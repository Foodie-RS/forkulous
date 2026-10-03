from __future__ import annotations

import logging
import math
from typing import Any, override

import pint

from api.models import (
    Density,
    ParserState,
    PintUnitCandidate,
    UnitDictResult,
    S1_UnitNames_Res,
)
from api.state import Provider, RequestState

PINT_EXCLUDE=["bag"]

class PintUnitNamesProvider[T](Provider[S1_UnitNames_Res]):
    units:dict[str,str]

    def __init__(self, ureg:pint.UnitRegistry[T]):
        super().__init__(_type=S1_UnitNames_Res)
        base_units = ['tonne', 'L', 'litre', 'gram', 'liter']
        prefixes = ["mega", "kilo", "milli", "micro","deci","centi"]
        prefixes_expanded = [it for p in prefixes for it in [""] + list(ureg._prefixes[p].aliases) + [ureg._prefixes[p].name]]
        self.units = {}
        for unit in base_units:
            for prefix in prefixes_expanded:
                self.units[f"{prefix}{unit}s"] = f"{prefix}{unit}"

    @override
    def execute(self, state: RequestState) -> S1_UnitNames_Res:
        return S1_UnitNames_Res(res=self.units)

class PintUnitProvider(Provider[UnitDictResult]):

    @override
    def deferred(self) -> bool:
        return True

    @override
    def execute(self, state:RequestState) -> UnitDictResult:
        logger = logging.getLogger("ingr_api").getChild("pint_provider")
        pstate = state.get(ParserState)
        if pstate.unit.is_none():
            logger.debug("No unit parsed, returning empty")
            return UnitDictResult(res={})
        unit = pstate.unit.unwrap()
        logger.debug(f"Looking up {unit}")
        ctx = pint.Context()
        ureg = state.get(pint.UnitRegistry[Any])
        try:
            pint_parsed = ureg.parse_units(unit, case_sensitive=False)
            logger.debug(f"Unit parsed successfully as {pint_parsed}")
            pint_amnt = 1 * pint_parsed
            for it in PINT_EXCLUDE:
                ctx.redefine(f"{it} = nan g")
            gram_compatible = pint_amnt.is_compatible_with("g", ctx)
            logger.debug(f"Unit is compatible with gram: {gram_compatible}")
            is_volume = False
            density_confidence:float|None = 1.0
            if not gram_compatible and pint_amnt.is_compatible_with("ml"):
                density_w = state.multi_get_first_opt(Density)
                if density_w.is_none():
                    logger.debug("No density available, skipping")
                    return UnitDictResult(res={})
                density = density_w.unwrap()
                density_confidence = density.confidence
                is_volume = True
                logger.debug(f"Adding density transformation: {density.density:.2f}")
                ctx.add_transformation("[volume]", "[mass]", lambda ureg, value, **kwargs: value * density.density * (ureg("g")/ureg("ml")))
                ctx.add_transformation("[length] ** 3", "[mass]", lambda ureg, value, **kwargs: value * density.density * (ureg("g")/ureg("ml")))
            if is_volume or gram_compatible:
                grams = float(pint_amnt.to("g", ctx).m)
                logger.debug(f"Calculating grams successful: {grams}")
                if not math.isnan(grams):
                    logger.debug(f"Adding unit candidate. Density confidence: {'not required' if gram_compatible else ('not present' if density is None else density.confidence)}")
                    cand = PintUnitCandidate(
                        comments=[],
                        entry_id=-1,
                        modifier=None,
                        gram_weight=grams,
                        plural_name=f"{pint_parsed}s",
                        singular_name=f"{pint_parsed}",
                        unit_name=f"{pint_parsed}",
                        density_confidence=density_confidence,
                        source="pint"
                    )
                    return UnitDictResult(
                        res={f"{pint_parsed}":[cand]}
                    )
        except pint.errors.UndefinedUnitError as e:
            logger.debug("Pint reported UndefinedUnitError:")
            logger.debug(e)
        logger.debug("Reporting no unit present.")
        return UnitDictResult(res={})
