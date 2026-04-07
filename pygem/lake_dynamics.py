"""
Python Glacier Evolution Model (PyGEM)

copyright © 2018 David Rounce <drounce@cmu.edu>

Distributed under the MIT license

Lake-terminating glacier dynamics module.
"""

import logging
import numpy as np
from oggm import cfg
from oggm.core.flowline import FluxBasedModel

log = logging.getLogger(__name__)


class LakeFluxBasedModel(FluxBasedModel):
    """
    FluxBasedModel adapted for existing lake-terminating glaciers.

    Replaces OGGM's ocean-style below_sl mass removal with sequential
    bin calving: the calving bucket fills via q = calving_k * d * h * w,
    and whole terminus bins are removed once the bucket can afford them.

    Water depth is computed at the calving front (last bin where
    surface_h > water_level and thick > 0 and bed_h < water_level).
    If no such bin exists (fully submerged tongue), the last ice bin is used.

    If the computed water depth d <= 0 (terminus bed is above water level),
    a fallback water level of moraine_elev - 20 m is used until a bin with
    bed_h < water_level is reached.

    Parameters
    ----------
    flowlines : list of oggm.Flowline
    moraine_elev : float
        Elevation of the terminal moraine [m a.s.l.], used for fallback
        water level when terminus bed is above the prescribed water_level.
    **kwargs : passed to FluxBasedModel (must include water_level)
    """

    def __init__(self, flowlines, moraine_elev=None, **kwargs):
        super().__init__(flowlines, **kwargs)
        self.moraine_elev = moraine_elev
        self._fallback_water_level = (
            moraine_elev - 20.0 if moraine_elev is not None else None
        )

    def step(self, dt):
        """Advance one timestep with lake-aware calving."""
        # Run parent SIA + MB step with calving disabled
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
            # thick > 0, and bed < water_level (i.e. partially submerged)
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
                    # Terminus bed is above water_level; use fallback
                    d, calving_front_idx = self._get_fallback_depth(fl)
                    if d is None:
                        continue
                    h = fl.thick[calving_front_idx]
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
                section[current] = 0
                current -= 1
                # Skip already-empty bins
                while current >= 0 and section[current] * fl.dx_meter <= 0:
                    current -= 1
                if current < 0:
                    break
                vol_terminus = section[current] * fl.dx_meter
                if vol_terminus <= 0:
                    break

            fl.section = section

        return dt_actual

    def _get_fallback_depth(self, fl):
        """
        When the terminus bed is above water_level, use moraine_elev - 20 m
        as a temporary water level to compute calving depth.

        Returns (d, bin_idx) or (None, None) if no suitable bin found.
        """
        if self._fallback_water_level is None:
            return None, None
        candidates = np.nonzero(
            (fl.surface_h > self._fallback_water_level)
            & (fl.thick > 0)
            & (fl.bed_h < self._fallback_water_level)
        )[0]
        if len(candidates) == 0:
            # Try last ice bin
            ice_bins = np.where(fl.thick > 0)[0]
            if len(ice_bins) == 0:
                return None, None
            idx = int(ice_bins[-1])
            h = fl.thick[idx]
            d = h - (fl.surface_h[idx] - self._fallback_water_level)
            return (d if d > 0 else None), idx
        idx = int(candidates[-1])
        h = fl.thick[idx]
        d = h - (fl.surface_h[idx] - self._fallback_water_level)
        return (d if d > 0 else None), idx