from collections.abc import Generator
import math
import numpy as np
import logging
from dataclasses import dataclass
from typing import Any, cast, override

from pandas.io.common import is_bool
from sentence_transformers import CrossEncoder, SentenceTransformer

from api.common import Option
from api.models import (
    EmptyUnitResult,
    FallbackUnitCandidate,
    IngredientCandidate,
    IngredientUnitCandidate,
    ParserState,
    S1_UnitNames_Res,
    UnitCandidate,
    S2_UnitDict_Res,
)
from api.providers.fdc_ingredient import FDCCandidates, LocalFDCIngredientProvider
from api.providers import unit_aggregation
from api.state import GeneratorProvider, OptionalProvider, Provider, RequestState

SERVING_MEASURE_UNITS = [1036,1049,1059,1069,1071,1096]
ITEM_MEASURE_UNITS = list(range(1013,1030)) + list(range(1031, 1038)) + list(range(1039,1049)) + list(range(1050,1059)) + [1060] + list(range(1063,1069)) + [1070] + list(range(1072,1121))
ITEM_NAMES=["fruit", "piece", ""]
MIN_UNIT_RESULTS=10
UNIT_INGR_CUTOFF=0.5

type _CorpusType=dict[str, list[dict[Any, Any]]]

def _collect_candidates(state:RequestState, corpus:_CorpusType) -> list[IngredientCandidate]:
    avail = [k for k in state.get_all(IngredientCandidate, only_use_available=True, from_provider=LocalFDCIngredientProvider) if k.score > UNIT_INGR_CUTOFF and k.id in corpus and len(corpus[k.id]) > 0]
    if len(avail) >= MIN_UNIT_RESULTS:
        avail.sort(key=lambda k:k.score, reverse=True)
        return avail
    for k in state.iter(IngredientCandidate, from_provider=LocalFDCIngredientProvider):
        if k in avail or k.score <= UNIT_INGR_CUTOFF or (k.id not in corpus) or (len(corpus[k.id]) == 0):
            continue
        avail.append(k)
        if len(avail) >= MIN_UNIT_RESULTS:
            break
    avail.sort(key=lambda k:k.score, reverse=True)
    return avail

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
    unit_corpus:_CorpusType

    def __init__(self,unit_corpus:dict[str, list[dict[Any, Any]]]):
        super().__init__(S1_UnitNames_Res)
        self.unit_corpus = unit_corpus

    @override
    def execute(self, state: RequestState) -> S1_UnitNames_Res:
        names:dict[str, str] = {}
        fdc_candidates = _collect_candidates(state, self.unit_corpus)
        for cand in fdc_candidates:
            units = self.unit_corpus[cand.id]
            for unit in units:
                if unit["unit_name"] != "":
                    names[unit["plural_name"]] = unit["singular_name"]
        return S1_UnitNames_Res(res=names)

@dataclass
class FDCUnitProvider(GeneratorProvider[UnitCandidate]):
    unit_corpus:_CorpusType
    unit_ranker:CrossEncoder
    use_sigmoid:bool=True

    def __init__(self, unit_corpus:_CorpusType, unit_ranker:CrossEncoder, use_sigmoid:bool=True):
        super().__init__(_type=UnitCandidate)
        self.unit_corpus = unit_corpus
        self.unit_ranker = unit_ranker
        self.use_sigmoid = use_sigmoid

    @override
    def execute(self, state: RequestState) -> Generator[UnitCandidate, None, None]:
        logger = logging.getLogger("ingr_api").getChild("FDCUnits")
        #TODO add case for fast=False
        rejects:list[IngredientUnitCandidate] = []
        pstate = state.get(ParserState)
        if pstate.unit.is_none_or(lambda k:len(k) == 0):
            logger.warning("ParserState.unit was empty, but FDCUnitProvider was called!")
            return
        unit_parse = pstate.unit.unwrap()
        for ingr in state.iter(IngredientCandidate, from_provider=LocalFDCIngredientProvider):
            if ingr.score <= UNIT_INGR_CUTOFF or (ingr.id not in self.unit_corpus) or (len(self.unit_corpus[ingr.id]) == 0):
                continue
            for unit in self.unit_corpus[ingr.id]:
                unit_cand = _unit(unit, ingr)
                if unit_parse.lower() in [k.lower() for k in [unit_cand.singular_name, unit_cand.plural_name, unit_cand.unit_name] if k is not None]:
                    logger.debug("Suitable unit found, yielding")
                    yield unit_cand
                else:
                    rejects.append(unit_cand)
        for it in rejects:
            logger.debug("Yielding CE-ranked units...")
            if it.unit_name is None or len(it.unit_name) > 0:
                continue
            logit_score = self.unit_ranker.predict((unit_parse, it.unit_name), convert_to_numpy=True)
            if self.use_sigmoid:
                logit_score = 1 / (1 + math.exp(-logit_score))
            score = float(logit_score)
            it.score_modifier = 0.5 + (0.5 * score)
            if it.relevance() > UNIT_INGR_CUTOFF:
                yield it

@dataclass
class LocalFDCUnitDictProv(Provider[S2_UnitDict_Res]):
    unit_corpus:_CorpusType

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

def _unit(unit_json:dict[str, Any], cand:IngredientCandidate) -> IngredientUnitCandidate:
    return IngredientUnitCandidate(
        measure_unit_id=unit_json["measure_unit_id"],
        ingredient_candidate=cand,
        comments=unit_json["comments"],
        modifier=unit_json["modifier"],
        entry_id=unit_json["entry_id"],
        gram_weight=unit_json["gram_weight"],
        plural_name=unit_json["plural_name"],
        singular_name=unit_json["singular_name"],
        seq_num=unit_json["seq_num"],
        unit_name=unit_json["unit_name"],
        source=unit_json["source"]
    )


def _get_default(crps:_CorpusType, state:RequestState) -> Option[UnitCandidate]:
    logger = logging.getLogger("ingr_api").getChild("_get_def_un")
    candidates = _collect_candidates(state, crps)
    logger.debug(f"Got {len(candidates)} candidates")
    for candidate in candidates:
        logger.debug(f"Looking up {candidate.description}")
        if str(candidate.id) in crps:
            units = list(crps[str(candidate.id)])
            if len(units) == 0:
                continue
            units.sort(key=lambda it:it["seq_num"])
            unit = units[0]
            logger.debug(f"Returning {unit['unit_name']} because it has the lowest seq_num")
            return Option.some(_unit(unit, candidate))
    logger.debug("No units, looking up `serving`")
    pstate = state.get(ParserState)
    if pstate.unit.is_some_and(lambda k:len(k) > 0):
        logger.debug("Unit not empty, not looking up serving to avoid infinite recursion")
        return Option.none()
    pstate_mod = ParserState(
        parsed_name=pstate.parsed_name,
        unit=Option.some("serving"),
        quantity=pstate.quantity,
        prep=pstate.prep
    )
    state_msk = state.mask()
    state_msk.set(ParserState, pstate_mod)
    candi = state_msk.get_optional(UnitCandidate, allow_parent=False) #TODO set from_provider to new provider
    state_msk.invalidate()
    if candi.is_none_or(lambda k: isinstance(k, FallbackUnitCandidate)):
        logger.debug("Refusing to use fallback unit candidate")
        return Option.none()
    candi_uw = candi.unwrap()
    if candi_uw.relevance() < 0.5:
        logger.debug("Score too low, returning None")
        return Option.none()
    else:
        return candi

@dataclass
class FDCEmptyUnitProv(OptionalProvider[EmptyUnitResult]):
    unit_corpus:_CorpusType

    def __init__(self, unit_corpus:_CorpusType):
        super().__init__(Option[EmptyUnitResult])
        self.unit_corpus = unit_corpus

    @override
    def execute(self, state:RequestState) -> Option[EmptyUnitResult]:
        res = _get_default(self.unit_corpus, state)
        return res.map(lambda res:EmptyUnitResult(res=res, confidence=res.relevance()))


@dataclass
class FDCFbUnitProv(OptionalProvider[FallbackUnitCandidate]):
    units:dict[str, FallbackUnitCandidate]
    name_eq:dict[str, str]

    def __init__(self, unit_corpus:_CorpusType):
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
