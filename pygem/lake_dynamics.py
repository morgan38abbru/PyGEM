"""
Python Glacier Evolution Model (PyGEM) - Lake Dynamics Module

Provides LakeFluxBasedModel, a subclass of OGGM's FluxBasedModel that
fixes two issues with using OGGM's marine calving scheme for proglacial lakes:

1. last_above_wl is found only among bins in the overdeepening (bed < water_level),
   preventing the search from jumping to the moraine or valley floor.

2. below_sl ice removal is constrained to bins contiguous with the calving front,
   and only proceeds if the bucket can fully afford it (no overflow into upstream bins).

Additionally implements an empirical area-depth calving scheme for newly-forming
proglacial lakes with a bin-by-bin volume-based progression:

EMPIRICAL PHASE (d_emp < d_actual):
- A_lake starts at the area of a single seed bin (the trigger bin).
- d_emp = 0.621 * A_lake^0.36 [m, m²] used as min(d_emp, d_actual).
- No below_sl removal — ice only disappears through the bucket loop.
- Bucket eats from the actual terminus upward bin by bin.
- Each bin eaten contributes to a volume counter; when the counter reaches
  the next target bin's volume, that bin is earned and A_lake grows.
- Target sequence: downstream from seed bin to terminus, then upstream.

STANDARD PHASE (d_emp >= d_actual):
- Switches permanently to standard calving law using d_actual.
- below_sl removal re-enables, but is bucket-limited:
  only as much submerged ice is removed as the bucket can pay for.
  If the bucket cannot fully cover the removal, it is skipped entirely
  that step (no overflow, no partial removal).
- Bucket then eats backward from last_above_wl bin by bin.

copyright © 2018 David Rounce <drounce@cmu.edu>

Distributed under the MIT license
"""

import logging
import numpy as np
from oggm import cfg
from oggm.core.flowline import FluxBasedModel

log = logging.getLogger(__name__)

# Empirical lake depth-area relationship
# depth [m] = LAKE_DEPTH_COEFF * area [m²] ^ LAKE_DEPTH_EXPONENT
LAKE_DEPTH_COEFF    = 0.621
LAKE_DEPTH_EXPONENT = 0.36


class LakeFluxBasedModel(FluxBasedModel):
    """
    FluxBasedModel adapted for proglacial-lake-terminating glaciers.

    Modifications to OGGM's calving scheme:

    1. last_above_wl search restricted to overdeepening (bed < water_level).
       When the entire tongue is submerged, the last ice bin is used as the
       effective front for computing q_calving.

    2. Empirical phase (use_empirical_depth=True, d_emp < d_actual):
       - No below_sl removal
       - Bucket eats from actual terminus upward, skipping empty bins
       - A_lake grows bin by bin as calving volume is earned
       - d_emp = 0.621 * A_lake^0.36 drives q_calving

    3. Standard phase (d_emp >= d_actual or use_empirical_depth=False):
       - below_sl removal re-enabled, but bucket-limited (skip if can't afford)
       - Bucket eats backward from last_above_wl bin by bin

    Parameters
    ----------
    flowlines : list of oggm.Flowline
    moraine_idx : int or None
        Upper bound on calving front retreat.
    use_empirical_depth : bool
        If True (default), use empirical area-depth ramp. False = standard law.
    initial_seed_bin : int or None
        Index of the bin that triggered lake formation.
    **kwargs : passed to FluxBasedModel
    """

    def __init__(self, flowlines, moraine_idx=None, use_empirical_depth=True,
                 initial_seed_bin=None, **kwargs):
        super().__init__(flowlines, **kwargs)
        self.moraine_idx         = moraine_idx
        self.use_empirical_depth = use_empirical_depth

        self._empirical_phase_complete = False

        fl = flowlines[0]

        # Seed A_lake with the trigger bin area
        if use_empirical_depth and initial_seed_bin is not None:
            seed_area = float(fl.widths_m[initial_seed_bin] * fl.dx_meter)
            self._lake_area_m2 = seed_area
            log.info(
                f'[LakeFluxBasedModel] Seeded at bin {initial_seed_bin}: '
                f'A={seed_area:.1f} m²  d_emp={self._empirical_depth():.2f} m'
            )
        else:
            self._lake_area_m2 = 0.0

        # Build bin sequence: downstream from seed bin to terminus,
        # then upstream from seed bin back toward the head.
        if use_empirical_depth and initial_seed_bin is not None:
            nx = fl.nx
            terminus_bins = np.where(fl.thick > 1.0)[0]
            terminus_idx  = int(terminus_bins[-1]) if len(terminus_bins) > 0 else nx - 1
            downstream    = list(range(initial_seed_bin + 1, terminus_idx + 1))
            upstream      = list(range(initial_seed_bin - 1, -1, -1))
            self._bin_sequence = downstream + upstream
        else:
            self._bin_sequence = []

        self._seq_idx            = 0
        self._calving_vol_earned = 0.0
        self._current_target_vol = 0.0

        if self._bin_sequence:
            self._set_next_target(fl)

        # Diagnostics
        self.diag_yr               = []
        self.diag_lake_area_m2     = []
        self.diag_d_emp            = []
        self.diag_d_actual         = []
        self.diag_d_used           = []
        self.diag_empirical_active = []
        self.diag_q_calving        = []
        self.diag_last_above_wl    = []

    def _set_next_target(self, fl):
        """Set _current_target_vol to the ice volume of the next target bin."""
        if self._seq_idx < len(self._bin_sequence):
            target_bin = self._bin_sequence[self._seq_idx]
            self._current_target_vol = float(fl.section[target_bin] * fl.dx_meter)
            log.debug(
                f'[LakeFluxBasedModel] Next target: bin {target_bin}  '
                f'vol={self._current_target_vol:.1f} m³'
            )
        else:
            self._current_target_vol = np.inf

    def _empirical_depth(self):
        """d_emp [m] from current A_lake [m²]. Returns 0 if not seeded."""
        if self._lake_area_m2 <= 0:
            return 0.0
        return LAKE_DEPTH_COEFF * (self._lake_area_m2 ** LAKE_DEPTH_EXPONENT)

    def step(self, dt):
        """Advance one step with lake-aware calving."""

        if getattr(self, '_lake_debug_entry', False):
            self._lake_step_count = getattr(self, '_lake_step_count', 0) + 1
            if self._lake_step_count <= 3:
                print(f'  [LakeFluxBasedModel.step] yr={self.yr:.4f} do_calving={self.do_calving}')

        # Run parent SIA + MB step with calving disabled
        was_calving = self.do_calving
        self.do_calving = False
        dt_actual = super().step(dt)
        self.do_calving = was_calving

        if not was_calving:
            return dt_actual

        for fl in self.fls:

            section = fl.section

            # ----------------------------------------------------------------
            # Find calving front for computing d_actual and q_calving
            # ----------------------------------------------------------------
            candidates = np.nonzero(
                (fl.surface_h > self.water_level) &
                (fl.thick > 0) &
                (fl.bed_h < self.water_level)
            )[0]

            if len(candidates) > 0:
                lawl_for_depth = candidates[-1]
                h        = fl.thick[lawl_for_depth]
                d_actual = h - (fl.surface_h[lawl_for_depth] - self.water_level)
                if d_actual <= 0 or h <= 0:
                    if getattr(self, '_lake_debug', False):
                        print(f'  yr={self.yr:.4f}: lawl={lawl_for_depth} SKIP d_actual<=0')
                    continue
            else:
                # Entire tongue submerged — use last ice bin for depth calc
                submerged_ice = np.where(
                    (fl.thick > 0) &
                    (fl.bed_h < self.water_level)
                )[0]
                if len(submerged_ice) == 0:
                    if getattr(self, '_lake_debug', False):
                        print(f'  yr={self.yr:.4f}: NO ice in overdeepening')
                    continue
                lawl_for_depth = int(submerged_ice[-1])
                h        = fl.thick[lawl_for_depth]
                d_actual = h  # fully submerged: use full thickness

            # ----------------------------------------------------------------
            # Choose calving depth
            # ----------------------------------------------------------------
            if self.use_empirical_depth and not self._empirical_phase_complete:
                d_emp = self._empirical_depth()
                if d_emp > 0 and d_emp >= d_actual:
                    self._empirical_phase_complete = True
                    d_used = d_actual
                    empirical_active = False
                    log.info(
                        f'[LakeFluxBasedModel] yr={self.yr:.2f}: '
                        f'd_emp ({d_emp:.2f} m) >= d_actual ({d_actual:.2f} m). '
                        f'Switching to standard calving law.'
                    )
                else:
                    d_used = d_emp if d_emp > 0 else d_actual
                    empirical_active = (d_emp > 0)
            else:
                d_emp = self._empirical_depth()
                d_used = d_actual
                empirical_active = False

            q_calving = self.calving_k * d_used * h * fl.widths_m[lawl_for_depth]

            if getattr(self, '_lake_debug', False):
                print(
                    f'  yr={self.yr:.4f}: lawl={lawl_for_depth} h={h:.1f} '
                    f'd_actual={d_actual:.1f} d_emp={d_emp:.2f} d_used={d_used:.1f} '
                    f'A_lake={self._lake_area_m2:.0f} m² '
                    f'emp_active={empirical_active} q={q_calving:.4e} '
                    f'bucket_before={fl.calving_bucket_m3:.4f}'
                )

            # Diagnostics
            self.diag_yr.append(float(self.yr))
            self.diag_lake_area_m2.append(float(self._lake_area_m2))
            self.diag_d_emp.append(float(d_emp))
            self.diag_d_actual.append(float(d_actual))
            self.diag_d_used.append(float(d_used))
            self.diag_empirical_active.append(bool(empirical_active))
            self.diag_q_calving.append(float(q_calving))
            self.diag_last_above_wl.append(int(lawl_for_depth))

            fl.calving_bucket_m3     += q_calving * dt_actual
            self.calving_m3_since_y0 += q_calving * dt_actual

            if section[lawl_for_depth] > 0:
                self.calving_rate_myr = q_calving / section[lawl_for_depth] * cfg.SEC_IN_YEAR

            max_idx = self.moraine_idx if self.moraine_idx is not None else fl.nx

            # ----------------------------------------------------------------
            # EMPIRICAL PHASE: no below_sl removal.
            # Bucket eats from actual terminus upward bin by bin.
            # ----------------------------------------------------------------
            if self.use_empirical_depth and not self._empirical_phase_complete:

                ice_bins = np.where(
                    (section > 0) &
                    (fl.bed_h < self.water_level) &
                    (np.arange(fl.nx) < max_idx)
                )[0]

                if len(ice_bins) == 0:
                    fl.section = section
                    continue

                terminus = int(ice_bins[-1])
                vol_last = section[terminus] * fl.dx_meter
                current  = terminus

                while fl.calving_bucket_m3 > vol_last and current > 0:
                    fl.calving_bucket_m3 -= vol_last

                    # Track earned volume for empirical bin progression
                    self._calving_vol_earned += vol_last
                    while (self._calving_vol_earned >= self._current_target_vol
                           and self._seq_idx < len(self._bin_sequence)):
                        self._calving_vol_earned -= self._current_target_vol
                        earned_bin = self._bin_sequence[self._seq_idx]
                        self._lake_area_m2 += float(
                            fl.widths_m[earned_bin] * fl.dx_meter
                        )
                        self._seq_idx += 1
                        self._set_next_target(fl)
                        log.debug(
                            f'[LakeFluxBasedModel] yr={self.yr:.2f}: '
                            f'Bin {earned_bin} earned. '
                            f'A_lake={self._lake_area_m2:.1f} m²  '
                            f'd_emp={self._empirical_depth():.2f} m'
                        )

                    section[current] = 0
                    current -= 1

                    # Skip empty bins
                    while current > 0 and section[current] * fl.dx_meter <= 0:
                        current -= 1

                    vol_last = section[current] * fl.dx_meter
                    if vol_last <= 0:
                        break

            # ----------------------------------------------------------------
            # STANDARD PHASE: below_sl removal re-enabled but bucket-limited.
            # Only remove submerged ice if the bucket can fully pay for it.
            # Then bucket eats backward from last_above_wl bin by bin.
            # ----------------------------------------------------------------
            else:
                # Only run standard phase if we have a valid last_above_wl
                # (i.e. not fully submerged)
                if len(candidates) == 0:
                    fl.section = section
                    continue

                last_above_wl = lawl_for_depth

                # Find contiguous below_sl bins adjacent to last_above_wl
                below_sl = np.zeros(fl.nx, dtype=bool)

                for i in range(last_above_wl + 1, max_idx):
                    if fl.surface_h[i] < self.water_level and fl.thick[i] > 0:
                        below_sl[i] = True
                    else:
                        break

                for i in range(last_above_wl - 1, -1, -1):
                    if fl.surface_h[i] < self.water_level and fl.thick[i] > 0:
                        below_sl[i] = True
                    else:
                        break

                # Bucket-limited below_sl removal: only proceed if the bucket
                # can fully pay for all the submerged ice.  No overflow into
                # upstream bins; if the bucket is short, skip removal this step.
                to_remove = np.sum(section[below_sl]) * fl.dx_meter
                if 0 < to_remove <= fl.calving_bucket_m3:
                    if getattr(self, '_lake_debug', False):
                        print(
                            f'  yr={self.yr:.4f}: STD below_sl removal '
                            f'to_remove={to_remove:.4f} m³ '
                            f'bucket={fl.calving_bucket_m3:.4f} m³ '
                            f'bins={np.where(below_sl)[0].tolist()}'
                        )
                    section[below_sl]     = 0
                    fl.calving_bucket_m3 -= to_remove
                elif to_remove > 0:
                    if getattr(self, '_lake_debug', False):
                        print(
                            f'  yr={self.yr:.4f}: STD below_sl SKIPPED '
                            f'(to_remove={to_remove:.4f} > bucket={fl.calving_bucket_m3:.4f})'
                        )

                # Bucket eats backward from last_above_wl
                vol_last = section[last_above_wl] * fl.dx_meter
                while fl.calving_bucket_m3 > vol_last and last_above_wl > 0:
                    fl.calving_bucket_m3 -= vol_last
                    section[last_above_wl] = 0
                    last_above_wl -= 1

                    vol_last = section[last_above_wl] * fl.dx_meter
                    if vol_last <= 0:
                        break

            fl.section = section

        return dt_actual