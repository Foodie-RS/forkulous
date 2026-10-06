import logging
from dataclasses import dataclass
from typing import Any, cast, override

from sentence_transformers import SentenceTransformer

from api.common import Option
from api.models import (
    EmptyUnitResult,
    FallbackUnitCandidate,
    IngredientUnitCandidate,
    ParserState,
    S1_UnitNames_Res,
    UnitCandidate,
    S2_UnitDict_Res,
)
from api.providers.fdc_ingredient import FDCCandidates
from api.providers import unit_aggregation
from api.state import OptionalProvider, Provider, RequestState

SERVING_MEASURE_UNITS = [1036,1049,1059,1069,1071,1096]
ITEM_MEASURE_UNITS = list(range(1013,1030)) + list(range(1031, 1038)) + list(range(1039,1049)) + list(range(1050,1059)) + [1060] + list(range(1063,1069)) + [1070] + list(range(1072,1121))
ITEM_NAMES=["fruit", "piece", ""]

def _cmp(items:list[tuple[Any, Any, bool]]):
    """
      Convenience method to compare multiple attributes of two arbitrary objects.
      Evaluates attributes in specified order, returning on the first attribute that is not equal.
    """
    for a,b, greater in items:
        if a and b:
            if (greater and (a > b)) or ((not greater) and (a < b)):
                return True
            elif (greater and (b > a)) or ((not greater) and (b < a)):
                return False
    return None

@dataclass
class FDCUnitNamesProv(Provider[S1_UnitNames_Res]):
    unit_corpus:dict[str, list[dict[Any, Any]]]

    def __init__(self,unit_corpus:dict[str, list[dict[Any, Any]]]):
        super().__init__(S1_UnitNames_Res)
        self.unit_corpus = unit_corpus

    @override
    def execute(self, state: RequestState) -> S1_UnitNames_Res:
        names:dict[str, str] = {}
        fdc_candidates = state.get(FDCCandidates)
        for cand in fdc_candidates.values():
            units = self.unit_corpus[cand.id]
            for unit in units:
                if unit["unit_name"] != "":
                    names[unit["plural_name"]] = unit["singular_name"]
        return S1_UnitNames_Res(res=names)

@dataclass
class LocalFDCUnitDictProv(Provider[S2_UnitDict_Res]):
    unit_corpus:dict[str, list[dict[Any, Any]]]

    def __init__(self, unit_corpus:dict[str, list[dict[Any,Any]]]):
        super().__init__(S2_UnitDict_Res)
        self.unit_corpus=unit_corpus
    @override
    def execute(self, state: RequestState) -> S2_UnitDict_Res:
        logger = logging.getLogger("ingr_api").getChild("FDCUnitProv")
        fdc_candidates = state.get(FDCCandidates)
        logger.debug(f"Got {len(fdc_candidates)} candidates")
        #TODO: replace generator with for loop
        def gener():
            for id,candidate in fdc_candidates.items():
                cand_units = self.unit_corpus[str(id)]
                for unit in cand_units:
                    new_unit = IngredientUnitCandidate(
                        measure_unit_id=unit["measure_unit_id"],
                        ingredient_candidate=candidate,
                        comments=unit["comments"],
                        modifier=unit["modifier"],
                        entry_id=unit["entry_id"],
                        gram_weight=unit["gram_weight"],
                        plural_name=unit["plural_name"],
                        singular_name=unit["singular_name"],
                        seq_num=unit["seq_num"],
                        unit_name=unit["unit_name"],
                        source=unit["source"]
                    )
                    yield new_unit


        units: dict[str, list[UnitCandidate]] = {}
        for new_unit in gener():
            added = False
            if new_unit.unit_name is not None and len(new_unit.unit_name) > 0:
                if new_unit.unit_name in units:
                    old_units = units[new_unit.unit_name]
                    added = False
                    for ix, unit in enumerate(old_units):
                        old_better = _cmp([
                            (unit.relevance(), new_unit.relevance(), True),
                            (cast(IngredientUnitCandidate, unit).seq_num, new_unit.seq_num, False),
                            (len(unit.comments), len(new_unit.comments), False),
                        ])
                        if not old_better:
                            old_units.insert(ix, new_unit)
                            added = True
                            break
                    if not added:
                        old_units.append(new_unit)
                else:
                    units[new_unit.unit_name] = [new_unit]
        logger.debug(f"Got a total of {len(units)} units.")
        return S2_UnitDict_Res(res=units)

def _get_default(unit_ranker:SentenceTransformer, crps:dict[str, list[dict[Any, Any]]], state:RequestState) -> Option[UnitCandidate]:
    logger = logging.getLogger("ingr_api").getChild("_get_def_un")
    cand = state.get(FDCCandidates)
    cnd_sorted = list(cand.values())
    cnd_sorted.sort(key=lambda k: k.score, reverse=True)
    logger.debug(f"Got {len(cnd_sorted)} candidates")
    for candidate in cnd_sorted:
        #TODO make this a static const variable
        if candidate.score < 0.2:
            continue
        logger.debug(f"Looking up {candidate.description}")
        if str(candidate.id) in crps:
            units = list(crps[str(candidate.id)])
            if len(units) == 0:
                continue
            units.sort(key=lambda it:it["seq_num"])
            unit = units[0]
            logger.debug(f"Returning {unit['unit_name']} because it has the lowest seq_num")
            return Option.some(IngredientUnitCandidate(
                measure_unit_id=unit["measure_unit_id"],
                ingredient_candidate=candidate,
                comments=unit["comments"],
                modifier=unit["modifier"],
                entry_id=unit["entry_id"],
                gram_weight=unit["gram_weight"],
                plural_name=unit["plural_name"],
                singular_name=unit["singular_name"],
                seq_num=unit["seq_num"],
                unit_name=unit["unit_name"],
                source=unit["source"]
            ))
    logger.debug("No units, looking up `serving`")
    #TODO maybe generalize this more? "serving" may be too narrow?
    res = unit_aggregation.find_unit_in_dict(state, "serving", unit_ranker)
    if len(res) == 0:
        logger.debug("No `serving` found.")
        return Option.none()
    elif res[0][1] < 0.5:
        logger.debug("Score too low, returning None")
        return Option.none()
    else:
        return Option.some(res[0][0])

@dataclass
class FDCEmptyUnitProv(OptionalProvider[EmptyUnitResult]):
    unit_corpus:dict[str, list[dict[Any, Any]]]
    unit_ranker:SentenceTransformer

    def __init__(self, unit_corpus:dict[str, list[dict[Any, Any]]], unit_ranker:SentenceTransformer):
        super().__init__(Option[EmptyUnitResult])
        self.unit_ranker = unit_ranker
        self.unit_corpus = unit_corpus

    @override
    def execute(self, state:RequestState) -> Option[EmptyUnitResult]:
        res = _get_default(self.unit_ranker, self.unit_corpus, state)
        return res.map(lambda res:EmptyUnitResult(res=res, confidence=res.relevance()))


@dataclass
class FDCFbUnitProv(OptionalProvider[FallbackUnitCandidate]):
    units:dict[str, FallbackUnitCandidate]
    name_eq:dict[str, str]

    def __init__(self, unit_corpus:dict[str, list[dict[Any, Any]]]):
        super().__init__(Option[FallbackUnitCandidate])
        self.name_eq = {}
        units: dict[str, list[dict[Any, Any]]]= {}
        self.units = {}
        for units_l in unit_corpus.values():
            for unit in units_l:
                unit_tgt = None
                for k in [unit["unit_name"], unit["singular_name"], unit["plural_name"]]:
                    if k in self.name_eq:
                        unit_tgt = self.name_eq[k]
                        break
                if unit_tgt is None:
                    unit_tgt = unit["unit_name"]
                    self.name_eq[unit["singular_name"]]= unit["unit_name"]
                    self.name_eq[unit["plural_name"]] = unit["unit_name"]
                lst = units.setdefault(unit_tgt, [])
                lst.append(unit)
        for k,v in units.items():
            grams = sum(it["gram_weight"] for it in v) / len(v)
            self.units[k] = FallbackUnitCandidate(
                unit_name=k,
                comments=[],
                entry_id=-1,
                gram_weight=grams,
                modifier=None,
                plural_name=v[0]["plural_name"],
                singular_name=v[0]["singular_name"],
                source="db_fallback"
            )

    @override
    def execute(self, state:RequestState) -> Option[FallbackUnitCandidate]:
        logger = logging.getLogger("ingr_api").getChild("FDCFbProv")
        pstate = state.get(ParserState)
        if pstate.unit.is_none_or(lambda k:k == ""):
            logger.debug("Can't provide fallback unit because unit is empty")
            return Option.none()
        unit = pstate.unit.unwrap()
        if unit in self.name_eq:
            res = Option.some_if(self.units.get(self.name_eq[unit]))
        else:
            res = Option.some_if(self.units.get(unit))
        if res.is_some():
            logger.debug("Found fallback in database")
        else:
            logger.debug("No fallback found")
        return res
