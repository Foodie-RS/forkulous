import logging
from typing import Any, override

from pint import UnitRegistry

from api.common import Option
from api.models import Density, UnitCandidate, S2_UnitDict_Res
from api.state import OptionalProvider, Provider, RequestState

GOOD_VOLUME_UNITS = {"cup", "cups", "cubic inch", "cubic inches", "quart", "quarts", "pint", "pints", "ml", "milliliter", "milliliters", "l", "liter", "liters"}

class UnitDensityProvider(OptionalProvider[Density]):

    def __init__(self):
        super().__init__(Option[Density])

    @override
    def execute(self, state: RequestState) -> Option[Density]:
        logger = logging.getLogger("ingr_api").getChild("DefDensProv")
        lst = state.get_all(S2_UnitDict_Res, allow_deferred=False)
        dct_merged:dict[str, list[UnitCandidate]] = {}
        for it in lst:
            for k,v in it.res.items():
                inner_lst:list[UnitCandidate] = dct_merged.setdefault(k, [])
                inner_lst.extend(v)
        if len(dct_merged) == 0:
            logger.debug("Unit dict empty, can't provide density")
            return Option.none()
        ureg = state.get(UnitRegistry[Any])
        ml = ureg("ml")
        g = ureg("g")
        intersect = GOOD_VOLUME_UNITS.intersection(dct_merged.keys())
        logger.debug("Good Volume Units: [%s]", ", ".join(intersect))
        if len(intersect) > 0:
            # get item with maximum relevance
            unit_name, unit = max([(k,outer[0]) for k,outer in dct_merged.items() if k in intersect], key=lambda a:a[1].relevance())
            pint_unit = ureg(unit_name)
            cnt = 1 * pint_unit
            density = (unit.gram_weight * g) / cnt
            density = density.to(g/ml)
            return Option.some(Density(
                confidence=unit.relevance(),
                density=float(density.m)
            ))
        else:
            logger.debug("No good volume units in unit dict")
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
