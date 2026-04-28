"""
Python Glacier Evolution Model (PyGEM)

copyright © 2018 David Rounce <drounce@cmu.edu>

Distributed under the MIT license

Lake-terminating glacier dynamics module.
"""

import logging
import numpy as np
from oggm import cfg
from oggm.core.flowline import SemiImplicitModel

log = logging.getLogger(__name__)

# Empirical lake depth-area scaling: depth [m] = COEFF * area [m^2] ^ EXP
LAKE_DEPTH_COEFF = 0.621
LAKE_DEPTH_EXPONENT = 0.36


class LakeSemiImplicitModel(SemiImplicitModel):
    """
    SemiImplicitModel adapted for existing lake-terminating glaciers.

    Replaces OGGM's ocean-style below_sl mass removal with sequential
    bin calving: the calving bucket fills via q = calving_k * d * h * w,
    and whole terminus bins are removed once the bucket can afford them.

    Water depth is computed at the calving front (last bin where
    surface_h > water_level and thick > 0 and bed_h < water_level).
    If no such bin exists (fully submerged tongue), the last ice bin is used.

    Parameters
    ----------
    flowlines : list of oggm.Flowline
    **kwargs : passed to SemiImplicitModel (must include water_level)
    """

    def __init__(self, flowlines, **kwargs):
        super().__init__(flowlines, **kwargs)
        self._lake_area_continuous = 0.0
        self._lake_volume_continuous = 0.0

    def step(self, dt):
        """Advance one timestep with lake-aware calving."""
        was_calving = self.do_calving
        self.do_calving = False
        dt_actual = super().step(dt)
        self.do_calving = was_calving

        if not was_calving:
            return dt_actual

        for fl in self.fls:
            section = fl.section

            # ----------------------------------------------------------
            # Find calving front: last bin where surface > water_level,
            # thick > 0, and bed < water_level (partially submerged)
            # ----------------------------------------------------------
            candidates = np.nonzero(
                (fl.surface_h > self.water_level)
                & (fl.thick > 0)
                & (fl.bed_h < self.water_level)
            )[0]

            if len(candidates) > 0:
                calving_front_idx = int(candidates[-1])
                h = fl.thick[calving_front_idx]
                d = h - (fl.surface_h[calving_front_idx] - self.water_level)
                if d <= 0 or h <= 0:
                    continue
            else:
                # Entire tongue submerged — use last ice bin
                ice_bins = np.where(
                    (fl.thick > 0) & (fl.bed_h < self.water_level)
                )[0]
                if len(ice_bins) == 0:
                    continue
                calving_front_idx = int(ice_bins[-1])
                h = fl.thick[calving_front_idx]
                d = h  # fully submerged: use full thickness

            # ----------------------------------------------------------
            # Fill calving bucket
            # ----------------------------------------------------------
            q_calving = (
                self.calving_k * d * h * fl.widths_m[calving_front_idx]
            )
            fl.calving_bucket_m3 += q_calving * dt_actual
            self.calving_m3_since_y0 += q_calving * dt_actual

            if section[calving_front_idx] > 0:
                self.calving_rate_myr = (
                    q_calving / section[calving_front_idx] * cfg.SEC_IN_YEAR
                )

            # ----------------------------------------------------------
            # Sequential bin removal: eat from terminus upward
            # ----------------------------------------------------------
            ice_bins_all = np.where(section > 0)[0]
            if len(ice_bins_all) == 0:
                fl.section = section
                continue

            terminus = int(ice_bins_all[-1])
            vol_terminus = section[terminus] * fl.dx_meter
            current = terminus

            while fl.calving_bucket_m3 >= vol_terminus and current >= 0:
                fl.calving_bucket_m3 -= vol_terminus
                bin_area = float(fl.widths_m[current] * fl.dx_meter)
                bin_depth = max(self.water_level - fl.bed_h[current], 0.0)
                self._lake_area_continuous += bin_area
                self._lake_volume_continuous += bin_area * bin_depth
                section[current] = 0
                current -= 1
                while current >= 0 and section[current] * fl.dx_meter <= 0:
                    current -= 1
                if current < 0:
                    break
                vol_terminus = section[current] * fl.dx_meter
                if vol_terminus <= 0:
                    break

            # Fractional credit for partial fill of the current front bin
            if current >= 0 and vol_terminus > 0:
                frac = min(fl.calving_bucket_m3 / vol_terminus, 1.0)
                bin_area = float(fl.widths_m[current] * fl.dx_meter)
                bin_depth = max(self.water_level - fl.bed_h[current], 0.0)
                self._lake_area_continuous += frac * bin_area
                self._lake_volume_continuous += frac * bin_area * bin_depth

            fl.section = section

        return dt_actual