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


# ---------------------------------------------------------------------------
# Shared supraglacial-lake growth-rate utilities.
#
# Used in two places that must stay consistent:
#   1. Here, to *backdate* a present-day observed lake-coverage raster to the
#      simulation start year.
#   2. In massbalance.py, to *grow* coverage forward year-by-year.
# ---------------------------------------------------------------------------

def load_lake_growth_rules(pygem_prms):
    """
    Load the slope-dependent annual lake growth-rate table.
    Expected CSV columns: slope_min_deg, slope_max_deg, growth_rate_annual
    Returns a list of (slope_min_deg, slope_max_deg, growth_rate_annual) tuples,
    or an empty list if the file can't be found/parsed.
    """
    import pandas as pd
    try:
        growth_fp = (
            pygem_prms['root']
            + pygem_prms['mb']['supra_lake_relpath']
            + pygem_prms['mb']['supra_lake_growth_fn']
        )
        df = pd.read_csv(growth_fp)
        return list(zip(
            df['slope_min_deg'].values,
            df['slope_max_deg'].values,
            df['growth_rate_annual'].values,
        ))
    except Exception:
        return []


def lake_overdeepening_mask(bed_h, thick):
    """Boolean mask (terminus-relative) of the overdeepening region a lake may grow in."""
    mask = np.zeros(len(bed_h), dtype=bool)
    terminus_bins = np.where(thick > 1.0)[0]
    if len(terminus_bins) == 0:
        return mask
    terminus_idx = int(terminus_bins[-1])
    moraine_elev = max(
        bed_h[terminus_idx],
        bed_h[terminus_idx + 1] if terminus_idx < len(bed_h) - 1 else bed_h[terminus_idx],
    )
    for i in range(terminus_idx, -1, -1):
        if bed_h[i] < moraine_elev:
            mask[i] = True
        else:
            break
    return mask


def lake_bin_slopes_deg(surface_h, dx_meter):
    """Forward-difference surface slope [deg] per bin; last bin copies second-to-last."""
    ror = np.zeros_like(surface_h, dtype=float)
    ror[:-1] = (surface_h[:-1] - surface_h[1:]) / dx_meter
    ror[-1] = ror[-2] if len(ror) > 1 else 0.0
    return np.degrees(np.arctan(np.abs(ror)))


def lookup_lake_growth_rate(slope_deg, growth_rules):
    """Annual growth rate for a given slope [deg]; 0.0 if no rule matches."""
    for s_min, s_max, r in growth_rules:
        if s_min <= slope_deg < s_max:
            return r
    return 0.0


def backdate_supra_lake_coverage(coverage, bed_h, surface_h, thick, dx_meter, growth_rules, n_years):
    """
    Back-project observed present-day coverage to n_years earlier by inverting
    cov_(t+1) = min(cov_t * (1+rate), 1.0).

    Bins currently at the 1.0 cap are left unchanged -- growth stops once a bin
    saturates, so how long ago that happened can't be recovered from the data.
    Bins outside the overdeepening, or whose slope matches no growth rule
    (rate == 0), are also left unchanged, since the forward model never grows
    them either.

    Returns (backdated_coverage, n_saturated_bins) for logging/QA.
    """
    backdated = coverage.copy()
    if n_years <= 0 or not growth_rules or not np.any(coverage > 0):
        return backdated, 0

    mask = lake_overdeepening_mask(bed_h, thick)
    slopes_deg = lake_bin_slopes_deg(surface_h, dx_meter)

    n_saturated = 0
    for bin_idx in np.where(coverage > 0)[0]:
        if not mask[bin_idx]:
            continue
        rate = lookup_lake_growth_rate(slopes_deg[bin_idx], growth_rules)
        if rate <= 0.0:
            continue
        if coverage[bin_idx] >= 1.0:
            n_saturated += 1
            continue
        backdated[bin_idx] = coverage[bin_idx] / ((1.0 + rate) ** n_years)

    return np.clip(backdated, 0.0, 1.0), n_saturated


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

            # The binned raster reflects present-day (observed) lake coverage,
            # not coverage at the simulation start year. Back-project it using
            # the same slope-dependent growth rules the forward model uses.
            SUPRA_LAKE_OBS_YEAR = 2026
            start_year = pygem_prms['climate']['sim_startyear']
            n_years = SUPRA_LAKE_OBS_YEAR - start_year
            growth_rules = load_lake_growth_rules(pygem_prms)
            supra_lake_binned_arr, n_saturated = backdate_supra_lake_coverage(
                supra_lake_binned_arr,
                bed_h=fl.bed_h,
                surface_h=fl.surface_h,
                thick=fl.thick,
                dx_meter=fl.dx_meter,
                growth_rules=growth_rules,
                n_years=n_years,
            )
            if n_saturated > 0:
                log.warning(
                    f'{gdir.rgi_id}: {n_saturated} supraglacial lake bin(s) were already at '
                    f'the 1.0 coverage cap in {SUPRA_LAKE_OBS_YEAR} -- left unchanged when backdating to '
                    f'{start_year} since their pre-saturation history cannot be recovered.'
                )

            fl.supra_lake = supra_lake_binned_arr

        else:
            fl.supra_lake = np.zeros(nbins)

        # Overwrite pickle
        gdir.write_pickle(flowlines, fl_str, filesuffix=filesuffix)

def load_lake_calving_data(pygem_prms, rgiid):
    """
    Check whether an RGI glacier has an entry in the lake calibration CSV, and
    classify it by status.

    Parameters
    ----------
    pygem_prms : dict
        PyGEM configuration dictionary
    rgiid : str
        RGI glacier ID string

    Returns
    -------
    dict or None
        Returns None if the glacier has no row in the CSV at all (i.e., it was
        never assessed -- falls through to the future-detection scheme).

        If found, returns:
        {'status': str,                      # 'existing_growing' | 'existing_nongrowing'
         'calving_k': float or None,          # None for existing_nongrowing
         'water_level': float or None,
         'moraine_elev': float or None}

        Note: 'existing_nongrowing' rows are expected to have calving_k as NaN;
        this is intentional (no calving applied) rather than an error condition.
    """
    import pandas as pd

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

    status = str(row['status']).strip() if 'status' in row and not pd.isna(row['status']) else None

    if status == 'existing_nongrowing':
        return {
            'status': 'existing_nongrowing',
            'calving_k': None,
            'water_level': float(row['water_level']) if not pd.isna(row['water_level']) else None,
            'moraine_elev': float(row['moraine_elev']) if 'moraine_elev' in row and not pd.isna(row['moraine_elev']) else None,
        }

    # status == 'existing_growing' (or missing/legacy rows without a status column)
    if pd.isna(row['calving_k']):
        # Row exists but has no calving_k and isn't explicitly marked nongrowing --
        # treat as not-yet-calibrated rather than silently defaulting to land-terminating.
        return None

    moraine_elev = float(row['moraine_elev']) if 'moraine_elev' in row and not pd.isna(row['moraine_elev']) else None

    return {
        'status': 'existing_growing',
        'calving_k': float(row['calving_k']),
        'water_level': float(row['water_level']),
        'moraine_elev': moraine_elev,
    }

def detect_lake_formation_potential(fls, threshold_depth=20.0, dry_tolerance_bins=2):
    """
    Detect whether a glacier has potential for proglacial lake formation.

    Identifies the terminal moraine as the highest bed elevation at or just
    downstream of the terminus (last bin with ice thickness > 1 m).
    The overdeepened zone is all contiguous bins upstream of the terminus
    whose bed elevation is below the moraine.

    The returned dict now includes 'overdeepened_area_km2': the total
    planimetric area (km²) of all overdeepened bins whose bed elevation is
    below the prescribed water level (moraine_elev - threshold_depth).
    Use this to gate lake formation on a minimum basin size.

    Parameters
    ----------
    fls : list of oggm.Flowline
    threshold_depth : float
        Depth below moraine crest at which calving activates [m] (default 20)

    Returns
    -------
    dict or None
        {'moraine_elevation': float,
         'lake_water_level': float,
         'overdeepened_bins': np.ndarray of int,
         'overdeepened_area_km2': float}
        Returns None if no overdeepening is found.
    """
    fl = fls[0]
    bed = fl.bed_h
    thickness = fl.thick

    terminus_bins = np.where(thickness > 1.0)[0]
    if len(terminus_bins) == 0:
        return None

    terminus_idx = int(terminus_bins[-1])
    if terminus_idx >= len(bed) - 1:
        return None

    moraine_elev = max(bed[terminus_idx], bed[terminus_idx + 1])

    ui = overdeepening_upstream_intersection(fl, moraine_elev, dry_tolerance_bins=dry_tolerance_bins)
    if ui['upstream_edge_idx'] is None:
        return None
    overdeepened_bins = np.array(ui['span_bin_indices'], dtype=int)

    water_level = moraine_elev - threshold_depth

    # Planimetric area of overdeepened bins whose bed is below the water level.
    # These are the bins that will actually be inundated; bins between the water
    # level and the moraine crest are above water and not counted.
    submerged_bins = overdeepened_bins[bed[overdeepened_bins] < water_level]
    if len(submerged_bins) > 0:
        overdeepened_area_m2 = float(
            np.sum(fl.widths_m[submerged_bins] * fl.dx_meter)
        )
    else:
        overdeepened_area_m2 = 0.0

    return {
        'moraine_elevation': float(moraine_elev),
        'lake_water_level': float(water_level),
        'overdeepened_bins': overdeepened_bins,
        'overdeepened_area_km2': overdeepened_area_m2 / 1e6,
    }

def overdeepening_upstream_intersection(fl, water_level, dry_tolerance_bins=2):
    """
    Two-phase upstream walk to find the extent of an overdeepened basin.

    Phase A: walk upstream from the terminus (last bin with thick > 1.0) to
    find the first WET bin (bed_h < water_level) -- the entry point. Handles
    the case where the terminus itself sits above water_level (e.g. a
    calibrated/data-driven water level on a glacier that has since retreated,
    or a terminus bin whose bed happens to sit at/above its own local moraine).
    Phase B: continue upstream through the wet region, tolerating brief dry
    interruptions (up to dry_tolerance_bins consecutive dry bins) rather than
    stopping at the first one, only halting at a sustained dry run -- the
    true upstream boundary of the basin.

    Ported from the calibration_suitability_and_calving_k notebook used to
    produce calibrated_calving_k.csv, so that the basin extent PyGEM resolves
    at simulation time is consistent with what the calibration assumed.

    Parameters
    ----------
    fl : oggm.Flowline
    water_level : float
        Threshold elevation [m a.s.l.] -- either a calibrated water level or
        a moraine elevation, depending on the caller.
    dry_tolerance_bins : int
        Consecutive dry bins the walk can bridge over before stopping.

    Returns
    -------
    dict
        {'terminus_idx': int or None,
         'upstream_edge_idx': int or None (None = no overdeepening found),
         'span_bin_indices': list of int, terminus-first (may include
             dry-tolerated bins -- callers needing only submerged bins
             should filter with bed_h < water_level),
         'n_dry_bins_in_span': int or None}
    """
    bed = fl.bed_h
    thick = fl.thick
    ice_bins = np.where(thick > 1.0)[0]
    if len(ice_bins) == 0:
        return {'terminus_idx': None, 'upstream_edge_idx': None,
                'span_bin_indices': [], 'n_dry_bins_in_span': None}

    t = int(ice_bins[-1])

    entry_idx = None
    for i in range(t, -1, -1):
        if bed[i] < water_level:
            entry_idx = i
            break

    if entry_idx is None:
        return {'terminus_idx': t, 'upstream_edge_idx': None,
                'span_bin_indices': [], 'n_dry_bins_in_span': None}

    upstream_edge = entry_idx
    i = entry_idx - 1
    pending_dry = []
    while i >= 0:
        if bed[i] < water_level:
            upstream_edge = i
            pending_dry = []
        else:
            pending_dry.append(i)
            if len(pending_dry) > dry_tolerance_bins:
                break
        i -= 1

    span = list(range(t, upstream_edge - 1, -1))
    n_dry = int(np.sum(bed[span] >= water_level))

    return {'terminus_idx': t, 'upstream_edge_idx': upstream_edge,
            'span_bin_indices': span, 'n_dry_bins_in_span': n_dry}