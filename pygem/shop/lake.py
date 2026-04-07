"""
Python Glacier Evolution Model (PyGEM)

copyright © 2018 David Rounce <drounce@cmu.edu>

Distributed under the MIT license
"""

import logging
import os
import warnings

import numpy as np
import rasterio
import xarray as xr
from oggm import cfg
from oggm.core.gis import rasterio_to_gdir
from oggm.utils import entity_task, ncDataset

# pygem imports
from pygem.setup.config import ConfigManager

# instantiate ConfigManager
config_manager = ConfigManager()
# read the config
pygem_prms = config_manager.read_config()

# Module logger
log = logging.getLogger(__name__)

# Register the 'supra_lake' name so OGGM knows where to find the file
if 'supra_lake' not in cfg.BASENAMES:
    cfg.BASENAMES['supra_lake'] = ('supra_lake.tif', 'Raster of supraglacial lake fractional coverage data')


@entity_task(log, writes=['supra_lake'])
def supra_lake_to_gdir(gdir, add_to_gridded=True):
    """Reproject the supraglacial lake fractional coverage file to the given glacier directory.

    Variables are exported as new files in the glacier directory.
    Reprojecting lake data from one map proj to another is done.
    We use average resampling to preserve fractional lake coverage.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        where to write the data
    add_to_gridded : bool
        whether to add the lake data to the gridded dataset
    """
    supra_lake_dir = (
        pygem_prms['root']
        + pygem_prms['mb']['supra_lake_relpath']
        + 'supra_lake_tifs/'
        + gdir.rgi_region
        + '/'
    )

    glac_str_nolead = str(int(gdir.rgi_region)) + '.' + gdir.rgi_id.split('-')[1].split('.')[1]

    # If supraglacial lake data exists, then write to glacier directory
    if os.path.exists(supra_lake_dir + glac_str_nolead + '_lake.tif'):
        supra_lake_fn = supra_lake_dir + glac_str_nolead + '_lake.tif'
    else:
        supra_lake_fn = None

    if supra_lake_fn is not None:
        rasterio_to_gdir(gdir, supra_lake_fn, 'supra_lake', resampling='average')

    if add_to_gridded and supra_lake_fn is not None:
        output_fn = gdir.get_filepath('supra_lake')

        # append the supraglacial lake data to the gridded dataset
        with rasterio.open(output_fn) as src:
            grids_file = gdir.get_filepath('gridded_data')
            with ncDataset(grids_file, 'a') as nc:
                # Mask values to glacier outline only
                glacier_mask = nc['glacier_mask'][:]
                data = src.read(1) * glacier_mask
                # Clip to valid fractional range [0, 1]
                data = np.clip(data, 0, 1).astype(np.float32)

                # Write data
                vn = 'supra_lake'
                if vn in nc.variables:
                    v = nc.variables[vn]
                else:
                    v = nc.createVariable(vn, 'f4', ('y', 'x'), zlib=True)
                v.units = '-'
                v.long_name = 'Supraglacial lake fractional coverage'
                v[:] = data


@entity_task(log, writes=['inversion_flowlines'])
def supra_lake_binned(gdir, fl_str='inversion_flowlines', filesuffix=''):
    """Bin supraglacial lake fractional coverage to flowlines.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        where to write the data
    fl_str : str
        The name of the flowline file to read. Default is 'inversion_flowlines'.
    filesuffix : str
        The filesuffix to use when reading the flowline file. Default is ''.
    """
    # Nominal glaciers will throw error, so make sure flowlines exist
    try:
        flowlines = gdir.read_pickle(fl_str, filesuffix=filesuffix)
        fl = flowlines[0]

        assert len(flowlines) == 1, 'Error: binning supraglacial lakes only works for single flowlines at present'

    except:
        flowlines = None

    if flowlines is not None:
        nbins = len(fl.dis_on_line)

        # Add binned supraglacial lake coverage to flowlines
        if os.path.exists(gdir.get_filepath('supra_lake')):
            ds = xr.open_dataset(gdir.get_filepath('gridded_data'))
            glacier_mask = ds['glacier_mask'].values
            topo = ds['topo_smoothed'].values
            supra_lake = ds['supra_lake'].values
            ds.close()

            # Only bin on-glacier values
            idx_glac = np.where(glacier_mask == 1)
            topo_onglac = topo[idx_glac]
            supra_lake_onglac = supra_lake[idx_glac]

            # Bin edges
            z_center = (fl.surface_h[0:-1] + fl.surface_h[1:]) / 2
            z_bin_edges = np.concatenate(
                (
                    np.array([topo[idx_glac].max() + 1]),
                    z_center,
                    np.array([topo[idx_glac].min() - 1]),
                )
            )

            # Loop over bins and calculate mean fractional lake coverage for each bin
            supra_lake_binned_arr = np.zeros(nbins)
            for nbin in np.arange(0, len(z_bin_edges) - 1):
                bin_max = z_bin_edges[nbin]
                bin_min = z_bin_edges[nbin + 1]
                bin_idx = np.where((topo_onglac < bin_max) & (topo_onglac >= bin_min))[0]
                # Lake coverage for on-glacier bins
                if len(bin_idx) > 0:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', category=RuntimeWarning)
                        supra_lake_binned_arr[nbin] = np.nanmean(supra_lake_onglac[bin_idx])
                # Bins below present-day glacier assumed to have no lakes
                else:
                    supra_lake_binned_arr[nbin] = 0

            fl.supra_lake = supra_lake_binned_arr

        else:
            fl.supra_lake = np.zeros(nbins)

        # Overwrite pickle
        gdir.write_pickle(flowlines, fl_str, filesuffix=filesuffix)

# ============================================================================
# EXISTING LAKE CALIBRATION DATA LOADER
# ============================================================================

def load_lake_calving_data(pygem_prms, rgiid):
    """
    Check whether an RGI glacier has a calibrated proglacial lake entry.

    Parameters
    ----------
    pygem_prms : dict
        PyGEM configuration dictionary
    rgiid : str
        RGI glacier ID string

    Returns
    -------
    dict or None
        If found: {'calving_k': float, 'water_level': float, 'moraine_elev': float or None}
        Returns None if no lake entry exists or calving_k is NaN.
    """
    import pandas as pd
    import os

    lake_fa_fp = (
        pygem_prms['root']
        + pygem_prms['calib']['data']['frontalablation']['frontalablation_relpath']
        + pygem_prms['calib']['data']['frontalablation']['lake_fa_cal_fn']
    )
    if not os.path.exists(lake_fa_fp):
        return None

    lake_fa_df = pd.read_csv(lake_fa_fp)
    if rgiid not in list(lake_fa_df['RGIId']):
        return None

    row = lake_fa_df.loc[lake_fa_df['RGIId'] == rgiid].iloc[0]
    if pd.isna(row['calving_k']):
        return None

    moraine_elev = float(row['moraine_elev']) if 'moraine_elev' in row and not pd.isna(row['moraine_elev']) else None

    return {
        'calving_k': float(row['calving_k']),
        'water_level': float(row['water_level']),
        'moraine_elev': moraine_elev,
    }