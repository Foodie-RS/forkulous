import logging
from typing import Any, override

from pint import UnitRegistry

from api.common import Option
from api.models import Density, NonemptyAmount, ParserState, UnitCandidate
from api.state import OptionalProvider, Provider, RequestState

GOOD_VOLUME_UNITS = {"cup", "cups", "cubic inch", "cubic inches", "quart", "quarts", "pint", "pints", "ml", "milliliter", "milliliters", "l", "liter", "liters"}
VOL_AMNTS = [NonemptyAmount(
    confidence=1.0,
    quantity=1.0,
    unit=unit
) for unit in GOOD_VOLUME_UNITS]

class UnitDensityProvider(OptionalProvider[Density]):

    def __init__(self):
        super().__init__(Option[Density])

    @override
    def execute(self, state: RequestState) -> Option[Density]:
        logger = logging.getLogger("ingr_api").getChild("DefDensProv")
        ureg = state.get(UnitRegistry[Any])
        ml = ureg("ml")
        g = ureg("g")
        pstate = state.get(ParserState)
        state_msk = state.mask()
        state_msk.set(ParserState, ParserState(
            parsed_name=pstate.parsed_name,
            prep=pstate.prep,
            nonempty_amounts=VOL_AMNTS,
            outer_amount=Option.none(),
            empty_amount=Option.none()
        ))
        for unit in state_msk.iter(UnitCandidate, resolve_deferred=False, use_parent=False):
            if unit.unit_name is not None and unit.unit_name.lower() in GOOD_VOLUME_UNITS:
                logger.debug(f"Found good volume unit: {unit.unit_name.lower()}")
                pint_unit = ureg(unit.unit_name)
                cnt = 1 * pint_unit
                density = (unit.gram_weight * g) / cnt
                density = density.to(g/ml)
                return Option.some(Density(
                    confidence=unit.relevance(),
                    density=float(density.m)
                ))
        state_msk.invalidate()
        logger.debug("No good volume units found!")
        return Option.none()

class FallbackDensityProvider(Provider[Density]):
    def __init__(self):
        super().__init__(_type=Density)
    @override
    def execute(self, state: RequestState) -> Density:
        logger = logging.getLogger("ingr_api").getChild("FbDenseProv")
        logger.debug("Falling back to density 0.77.")
        return Density(
            density=0.77,
            confidence=-1.0
        )
