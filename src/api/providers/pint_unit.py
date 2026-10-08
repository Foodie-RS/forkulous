from __future__ import annotations

import logging
import math
from typing import Any, override

import pint

from api.common import Option
from api.models import (
    Density,
    ParserState,
    PintUnitCandidate,
    S1_UnitNames_Res,
    S2_UnitDict_Res,
    UnitCandidate,
)
from api.state import OptionalProvider, Provider, RequestState

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

def _candidate(grams:float, unit:pint.Unit, density_confidence:Option[float]) -> PintUnitCandidate:
    return PintUnitCandidate(
        comments=[],
        entry_id=-1,
        modifier=None,
        gram_weight=grams,
        plural_name=f"{unit}s",
        singular_name=f"{unit}",
        unit_name=f"{unit}",
        density_confidence=density_confidence.unwrap_or_union(None),
        source="pint"
    )

class PintUnitProvider(OptionalProvider[UnitCandidate]):

    def __init__(self):
        super().__init__(Option[UnitCandidate])

    @override
    def deferred(self) -> bool:
        return True

    @override
    def execute(self, state:RequestState) -> Option[UnitCandidate]:
        logger = logging.getLogger("ingr_api").getChild("pint_provider")
        pstate = state.get(ParserState)
        if len(pstate.unit) == 0:
            logger.debug("No unit parsed, returning empty")
            return Option.none()
        ureg = state.get(pint.UnitRegistry[Any])
        ctx = pint.Context()
        for it in PINT_EXCLUDE:
            ctx.redefine(f"{it} = nan g")
        units_parsed = [ureg.parse_units(unit, case_sensitive=False) for unit in pstate.unit]
        units_gram_compat:list[pint.Unit] = [u for u in units_parsed if u.is_compatible_with("g", ctx)]
        for unit in units_gram_compat:
            logger.debug(f"Attempting {unit} (gram compatible)")
            pint_amnt = 1 * unit
            try:
                grams = float(pint_amnt.m)
                if math.isnan(grams):
                    logger.debug("Unit was excluded")
                    continue
                logger.debug(f"Good unit: {unit}")
                return Option.some(_candidate(grams, unit, density_confidence=Option.some(1.0)))
            except pint.errors.UndefinedUnitError:
                pass
        density_w = state.get_optional(Density)
        if density_w.is_none():
            logger.debug("No gram-compatible units found and Density was None")
            return Option.none()
        density = density_w.unwrap()
        logger.debug(f"Adding density transformation: {density.density:.2f}")
        ctx.add_transformation("[volume]", "[mass]", lambda ureg, value, **kwargs: value * density.density * (ureg("g")/ureg("ml")))
        ctx.add_transformation("[length] ** 3", "[mass]", lambda ureg, value, **kwargs: value * density.density * (ureg("g")/ureg("ml")))
        density_confidence = density.confidence
        for unit_parsed in [k for k in units_parsed if k not in units_gram_compat]:
            logger.debug(f"Attempting {unit_parsed} (not gram compatible)")
            try:
                pint_amnt = 1 * unit_parsed
                if pint_amnt.is_compatible_with("ml"):
                    grams = float(pint_amnt.to("g", ctx).m)
                    logger.debug(f"Calculating grams successful: {grams}")
                    if math.isnan(grams):
                        logger.debug("Unit was excluded")
                        continue
                    logger.debug(f"Adding unit candidate. Density confidence: {density.confidence}")
                    return Option.some(_candidate(grams, unit_parsed, Option.some_if(density_confidence)))
            except pint.errors.UndefinedUnitError as e:
                logger.debug("Pint reported UndefinedUnitError:")
                logger.debug(e)
        logger.debug("Reporting no unit present.")
        return Option.none()
