"""
Python Glacier Evolution Model (PyGEM) - Lake Dynamics Module

copyright © 2018 David Rounce <drounce@cmu.edu>

Distributed under the MIT license

Provides LakeFluxBasedModel, a subclass of OGGM's FluxBasedModel adapted
for proglacial-lake-terminating glaciers.

OGGM's standard marine calving scheme has two behaviours that are physically
incorrect for proglacial lakes:

1. It searches for the last bin above water level across the entire flowline,
   which can jump over the moraine onto the valley floor and produce a
   spuriously deep calving front.

2. After computing the calving flux it immediately removes all ice below the
   water level ('below_sl removal'), which is not appropriate for lakes
   because submerged ice should be eaten progressively by the calving bucket
   rather than wiped instantly.

LakeFluxBasedModel fixes both:

1. The effective calving front is found only among bins where the bed lies
   below the water level (the lake basin). This prevents the search from
   escaping across the moraine onto the downstream valley floor.

2. The below_sl removal step is skipped entirely. The calving bucket removes
   ice bin by bin starting at the most downstream ice bin in the lake basin
   and working upstream, stopping at moraine_idx.
"""

import logging
import numpy as np
from oggm import cfg
from oggm.core.flowline import FluxBasedModel

log = logging.getLogger(__name__)


class LakeFluxBasedModel(FluxBasedModel):
    """FluxBasedModel adapted for proglacial-lake-terminating glaciers.

    The calving front is identified as the last bin where the surface is above
    the water level AND the bed is below the water level (partially emergent
    ice in the lake basin). If no such bin exists (tongue fully submerged),
    the last ice bin with bed below water level is used.

    The calving bucket removes ice starting at the terminus of the lake basin
    and working upstream, stopping at moraine_idx. No ice is removed
    instantaneously below the water level.

    Parameters
    ----------
    flowlines : list
        Glacier flowlines (single flowline only).
    moraine_idx : int or None
        Index of the bin just upstream of the lake basin (the moraine crest).
        Ice removal is constrained to bins with index < moraine_idx.
        If None, removal is unconstrained.
    **kwargs
        All other keyword arguments passed directly to FluxBasedModel,
        including y0, mb_model, glen_a, fs, is_tidewater, water_level,
        calving_k, etc.
    """

    def __init__(self, flowlines, moraine_idx=None, **kwargs):
        super().__init__(flowlines, **kwargs)
        self.moraine_idx = moraine_idx

    def step(self, dt):
        """Advance one time step with lake-appropriate calving.

        The parent FluxBasedModel.step() handles ice dynamics and mass
        balance with calving disabled. Lake calving is then applied
        separately using the corrected front geometry and without the
        below_sl removal.
        """
        # Run standard dynamics step with calving disabled
        was_calving = self.do_calving
        self.do_calving = False
        dt_actual = super().step(dt)
        self.do_calving = was_calving

        if not was_calving:
            return dt_actual

        for fl in self.fls:
            section = fl.section

            # Find the effective calving front:
            # prefer bins where surface is above water level but bed is below
            # (partially emergent ice — has both freeboard and submerged depth)
            candidates = np.nonzero(
                (fl.surface_h > self.water_level)
                & (fl.thick > 0)
                & (fl.bed_h < self.water_level)
            )[0]

            if len(candidates) > 0:
                # Use the most downstream emergent bin in the lake basin
                lawl = int(candidates[-1])
                h = fl.thick[lawl]
                d = h - (fl.surface_h[lawl] - self.water_level)
                if d <= 0 or h <= 0:
                    continue
            else:
                # Entire tongue submerged — use last ice bin with bed below water
                submerged = np.where(
                    (fl.thick > 0) & (fl.bed_h < self.water_level)
                )[0]
                if len(submerged) == 0:
                    continue
                lawl = int(submerged[-1])
                h = fl.thick[lawl]
                d = h

            # Standard k-calving law: q = k * d * h * width  [m3 s-1]
            q_calving = self.calving_k * d * h * fl.widths_m[lawl]

            fl.calving_bucket_m3 += q_calving * dt_actual
            self.calving_m3_since_y0 += q_calving * dt_actual

            if section[lawl] > 0:
                self.calving_rate_myr = q_calving / section[lawl] * cfg.SEC_IN_YEAR

            # Find removable ice: bins with ice, bed below water level,
            # and upstream of the moraine
            max_idx = self.moraine_idx if self.moraine_idx is not None else fl.nx
            ice_bins = np.where(
                (section > 0)
                & (fl.bed_h < self.water_level)
                & (np.arange(fl.nx) < max_idx)
            )[0]

            if len(ice_bins) == 0:
                fl.section = section
                continue

            # Remove ice from the terminus of the lake basin upward
            current = int(ice_bins[-1])
            vol_current = section[current] * fl.dx_meter

            while fl.calving_bucket_m3 > vol_current and current > 0:
                fl.calving_bucket_m3 -= vol_current
                section[current] = 0
                current -= 1
                # Skip already-empty bins
                while current > 0 and section[current] == 0:
                    current -= 1
                vol_current = section[current] * fl.dx_meter
                if vol_current <= 0:
                    break

            fl.section = section

        return dt_actual