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

# Empirical lake depth–area scaling: depth [m] = COEFF * area [m^2] ^ EXP
# From Shugar et al. (2020) or similar
LAKE_DEPTH_COEFF = 0.621
LAKE_DEPTH_EXPONENT = 0.36

class LakeFluxBasedModel(FluxBasedModel):
    """
    FluxBasedModel adapted for existing lake-terminating glaciers.

    Replaces OGGM's ocean-style below_sl mass removal with sequential
    bin calving: the calving bucket fills via q = calving_k * d * h * w,
    and whole terminus bins are removed once the bucket can afford them.

    Water depth is computed at the calving front (last bin where
    surface_h > water_level and thick > 0 and bed_h < water_level).
    If no such bin exists (fully submerged tongue), the last ice bin is used.

    If moraine_elev is provided and the terminus bed is above the prescribed
    water_level, the water level is permanently set to moraine_elev - 20 m.

    Parameters
    ----------
    flowlines : list of oggm.Flowline
    moraine_elev : float
        Elevation of the terminal moraine [m a.s.l.], used to override
        water_level when the terminus bed is above it.
    **kwargs : passed to FluxBasedModel (must include water_level)
    """

    def __init__(self, flowlines, moraine_elev=None, **kwargs):
        super().__init__(flowlines, **kwargs)
        self.moraine_elev = moraine_elev

        # If the terminus bed is above water_level, permanently use moraine - 20 m
        if moraine_elev is not None:
            fl = flowlines[0]
            ice_bins = np.where(fl.thick > 0)[0]
            if len(ice_bins) > 0:
                terminus_bed = fl.bed_h[int(ice_bins[-1])]
                if terminus_bed >= self.water_level:
                    self.water_level = moraine_elev - 20.0

        self._lake_area_continuous = 0.0    # running proglacial lake area [m^2]
        self._lake_volume_continuous = 0.0  # running proglacial lake volume [m^3]

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
                # Full bin removed: credit its entire area and volume
                bin_area = float(fl.widths_m[current] * fl.dx_meter)
                bin_depth = max(self.water_level - fl.bed_h[current], 0.0)
                self._lake_area_continuous += bin_area
                self._lake_volume_continuous += bin_area * bin_depth
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

            # Fractional credit for partial fill of the current front bin
            if current >= 0 and vol_terminus > 0:
                frac = min(fl.calving_bucket_m3 / vol_terminus, 1.0)
                bin_area = float(fl.widths_m[current] * fl.dx_meter)
                bin_depth = max(self.water_level - fl.bed_h[current], 0.0)
                self._lake_area_continuous += frac * bin_area
                self._lake_volume_continuous += frac * bin_area * bin_depth

            fl.section = section

        return dt_actual

    
class NewLakeFluxBasedModel(LakeFluxBasedModel):
    """
    LakeFluxBasedModel with empirical area-depth ramp-up for newly formed lakes.

    EMPIRICAL PHASE (active until d_emp >= d_actual):
    - A_lake is seeded with the area of the trigger bin (initial_seed_bin).
    - d_emp = LAKE_DEPTH_COEFF * A_lake^LAKE_DEPTH_EXPONENT
    - q_calving uses d_emp instead of true water depth
    - Bucket eats sequentially from the terminus upward
    - Each bin removed earns its area into A_lake, growing d_emp
    - No separate below_sl removal during this phase

    STANDARD PHASE (d_emp >= d_actual):
    - Permanently switches to LakeFluxBasedModel sequential calving
      using true water depth d_actual

    Parameters
    ----------
    flowlines : list of oggm.Flowline
    initial_seed_bin : int
        Index of the bin that triggered lake formation
    moraine_elev : float
        Moraine elevation used to set water_level if terminus bed is above
        the prescribed water_level [m a.s.l.]
    **kwargs : passed to LakeFluxBasedModel (must include water_level)
    """

    def __init__(self, flowlines, initial_seed_bin=None, moraine_elev=None,
                 **kwargs):
        super().__init__(flowlines, moraine_elev=moraine_elev, **kwargs)

        fl = flowlines[0]
        self._empirical_phase_complete = False

        # Seed lake area with trigger bin
        if initial_seed_bin is not None:
            self._lake_area_m2 = float(
                fl.widths_m[initial_seed_bin] * fl.dx_meter
            )
        else:
            self._lake_area_m2 = 0.0

        # Build ordered bin sequence for area accumulation:
        # downstream from seed to terminus, then upstream from seed to head
        if initial_seed_bin is not None:
            terminus_bins = np.where(fl.thick > 1.0)[0]
            terminus_idx = int(terminus_bins[-1]) if len(terminus_bins) > 0 else fl.nx - 1
            downstream = list(range(initial_seed_bin + 1, terminus_idx + 1))
            upstream = list(range(initial_seed_bin - 1, -1, -1))
            self._bin_sequence = downstream + upstream
        else:
            self._bin_sequence = []

        self._seq_idx = 0
        self._calving_vol_earned = 0.0
        self._current_target_vol = 0.0
        if self._bin_sequence:
            self._set_next_target(fl)

    def _empirical_depth(self):
        """d_emp [m] from current A_lake [m^2]."""
        if self._lake_area_m2 <= 0:
            return 0.0
        return LAKE_DEPTH_COEFF * (self._lake_area_m2 ** LAKE_DEPTH_EXPONENT)

    def _set_next_target(self, fl):
        """Set the ice volume of the next bin to be earned into the lake."""
        if self._seq_idx < len(self._bin_sequence):
            target_bin = self._bin_sequence[self._seq_idx]
            self._current_target_vol = float(
                fl.section[target_bin] * fl.dx_meter
            )
        else:
            self._current_target_vol = np.inf

    def step(self, dt):
        """Advance one timestep, using empirical depth until d_emp >= d_actual."""

        # Run parent SIA + MB step with calving disabled
        was_calving = self.do_calving
        self.do_calving = False
        dt_actual = super(LakeFluxBasedModel, self).step(dt)
        self.do_calving = was_calving

        if not was_calving:
            return dt_actual

        for fl in self.fls:
            section = fl.section

            # ----------------------------------------------------------
            # Compute true water depth at calving front
            # ----------------------------------------------------------
            candidates = np.nonzero(
                (fl.surface_h > self.water_level)
                & (fl.thick > 0)
                & (fl.bed_h < self.water_level)
            )[0]

            if len(candidates) > 0:
                calving_front_idx = int(candidates[-1])
                h = fl.thick[calving_front_idx]
                d_actual = h - (fl.surface_h[calving_front_idx] - self.water_level)
                if d_actual <= 0 or h <= 0:
                    d_actual_valid = False
                else:
                    d_actual_valid = True
            else:
                ice_bins = np.where(
                    (fl.thick > 0) & (fl.bed_h < self.water_level)
                )[0]
                if len(ice_bins) == 0:
                    # No ice in overdeepening — try fallback
                    ice_bins = np.where(fl.thick > 0)[0]
                    if len(ice_bins) == 0:
                        continue
                calving_front_idx = int(ice_bins[-1] if len(ice_bins) > 0 else 0)
                h = fl.thick[calving_front_idx]
                d_actual = h
                d_actual_valid = h > 0

            if not d_actual_valid:
                continue

            # ----------------------------------------------------------
            # Choose calving depth: empirical or actual
            # ----------------------------------------------------------
            if not self._empirical_phase_complete:
                d_emp = self._empirical_depth()
                if d_emp > 0 and d_emp >= d_actual:
                    # Permanently switch to standard phase
                    self._empirical_phase_complete = True
                    log.info(
                        f'yr={self.yr:.2f}: d_emp={d_emp:.2f} >= '
                        f'd_actual={d_actual:.2f}. Switching to standard calving.'
                    )
                    d_used = d_actual
                    use_empirical = False
                else:
                    d_used = d_emp if d_emp > 0 else d_actual
                    use_empirical = d_emp > 0
            else:
                d_used = d_actual
                use_empirical = False

            # ----------------------------------------------------------
            # Fill calving bucket
            # ----------------------------------------------------------
            q_calving = (
                self.calving_k * d_used * h * fl.widths_m[calving_front_idx]
            )
            fl.calving_bucket_m3 += q_calving * dt_actual
            self.calving_m3_since_y0 += q_calving * dt_actual

            if section[calving_front_idx] > 0:
                self.calving_rate_myr = (
                    q_calving / section[calving_front_idx] * cfg.SEC_IN_YEAR
                )

            # ----------------------------------------------------------
            # Remove bins: empirical phase vs standard phase
            # ----------------------------------------------------------
            if use_empirical:
                # Eat from actual terminus upward;
                # accumulate earned volume to grow A_lake
                ice_bins_all = np.where(
                    (section > 0) & (fl.bed_h < self.water_level)
                )[0]
                if len(ice_bins_all) == 0:
                    fl.section = section
                    continue

                terminus = int(ice_bins_all[-1])
                vol_terminus = section[terminus] * fl.dx_meter
                current = terminus

                while fl.calving_bucket_m3 >= vol_terminus and current >= 0:
                    fl.calving_bucket_m3 -= vol_terminus
                    self._calving_vol_earned += vol_terminus

                    # Check if we have earned the next bin in the sequence
                    while (
                        self._calving_vol_earned >= self._current_target_vol
                        and self._seq_idx < len(self._bin_sequence)
                    ):
                        self._calving_vol_earned -= self._current_target_vol
                        earned_bin = self._bin_sequence[self._seq_idx]
                        self._lake_area_m2 += float(
                            fl.widths_m[earned_bin] * fl.dx_meter
                        )
                        self._seq_idx += 1
                        self._set_next_target(fl)

                    # Full bin removed: credit area and volume
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

            else:
                # Standard phase: same sequential calving as LakeFluxBasedModel
                ice_bins_all = np.where(section > 0)[0]
                if len(ice_bins_all) == 0:
                    fl.section = section
                    continue

                terminus = int(ice_bins_all[-1])
                vol_terminus = section[terminus] * fl.dx_meter
                current = terminus

                while fl.calving_bucket_m3 >= vol_terminus and current >= 0:
                    fl.calving_bucket_m3 -= vol_terminus
                    # Full bin removed: credit area and volume
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