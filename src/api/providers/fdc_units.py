import logging
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any, override

import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search
from torch._functorch.vmap import in_dims_t

from api.common import Option
from api.models import (
    EmptyUnitResult,
    FallbackUnitCandidate,
    IngredientCandidate,
    IngredientUnitCandidate,
    NonemptyAmount,
    ParserState,
    S1_UnitNames_Res,
    UnitCandidate,
)
from api.providers.fdc_ingredient import LocalFDCIngredientProvider
from api.state import GeneratorProvider, OptionalProvider, Provider, RequestState

ITEM_NAMES={"fruit", "whole","medium","regular", "serving"}
SKIP_NAMES={"cup", "ounce", "oz", "fl oz"}
MIN_UNIT_RESULTS=3
UNIT_INGR_CUTOFF=0.5
MAX_INGREDIENT_PULL_COUNT=10

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
                if any(k is None or k == "" for k in [unit["unit_name"], unit["singular_name"], unit["plural_name"]]):
                    continue
                names[unit["plural_name"]] = unit["singular_name"]
        return S1_UnitNames_Res(res=names)

@dataclass
class FDCUnitProvider(GeneratorProvider[UnitCandidate]):
    unit_corpus:_CorpusType
    unit_ranker:SentenceTransformer
    use_sigmoid:bool=True

    def __init__(self, unit_corpus:_CorpusType, unit_ranker:SentenceTransformer, use_sigmoid:bool=True):
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
        unit_names = [amnt.unit for amnt in pstate.nonempty_amounts]
        logger.debug(f"FDCUnitsProvider called for units {unit_names}")
        if len(unit_names)==0:
            logger.warning("ParserState.unit was empty, but FDCUnitProvider was called!")
            return
        unit_set = {p for p in unit_names}
        ingr_pull_count = 0
        for ingr in state.iter(IngredientCandidate, from_provider=LocalFDCIngredientProvider):
            if ingr.score <= UNIT_INGR_CUTOFF:
                logger.debug(f"Score for {ingr.description} is too low")
                continue
            if (ingr.id not in self.unit_corpus) or (len(self.unit_corpus[ingr.id]) == 0):
                logger.debug(f"No units for {ingr.description}")
                continue
            ingr_pull_count += 1
            for unit_parsed in self.unit_corpus[ingr.id]:
                unit_cand = _unit(unit_parsed, ingr)
                names = [k.lower() for k in [unit_cand.singular_name, unit_cand.plural_name, unit_cand.unit_name] if k is not None]
                intersect = unit_set.intersection(names)
                if len(intersect) > 0:
                    logger.debug(f"Suitable unit found: {unit_cand.unit_name}, yielding")
                    cand_ix,_ = max((k for k in enumerate(pstate.nonempty_amounts) if k[1].unit in intersect), key=lambda it:it[1].confidence)
                    unit_cand.parsed_from=Option.some(cand_ix)
                    yield unit_cand
                else:
                    rejects.append(unit_cand)
            if ingr_pull_count >= MAX_INGREDIENT_PULL_COUNT:
                break
        embd_query = self.unit_ranker.encode(unit_names, convert_to_tensor=True)
        docs:dict[str, UnitCandidate] = {}
        for it in rejects:
            #TODO maybe consider respecting comments as well? Although this might not be so important as these are already rejects
            cont_outer = False
            names = [k.lower() for k in [it.unit_name, it.singular_name, it.plural_name] if k is not None and len(k) > 0]
            for k in names:
                if k in docs:
                    cont_outer = True
                    docs[k] = max([it, docs[k]], key=lambda inner:inner.relevance())
                    break
            if not cont_outer and len(names) > 0:
                docs[names[0]] = it
        docs_list:list[UnitCandidate] = list(docs.values())
        #TODO Create index over docs? They are static data
        embd_docs = self.unit_ranker.encode([(k.unit_name or k.singular_name or k.plural_name) for k in docs_list], convert_to_tensor=True)
        results = semantic_search(embd_query, embd_docs, top_k=10)
        results_match:list[tuple[int, float, UnitCandidate]] = [(ix, it["score"], docs_list[it["corpus_id"]]) for ix, outer in enumerate(results) for it in outer]
        results_match.sort(key=lambda k:k[1], reverse=True)
        for parsed, score, candidate in results_match:
            # candidates can appear more than once
            candidate = candidate.clone()
            # penalty for low semantic similarity
            candidate.score_modifier = 0.5 + (0.5 * score)
            candidate.parsed_from=Option.some(parsed)
            if candidate.relevance() > UNIT_INGR_CUTOFF:
                logger.debug(f"Yielding semantic-searched unit {candidate.unit_name} (BE score: {score}, new total score {candidate})...")
                yield candidate

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
            rem:list[int] = []
            for ix,unit in enumerate(units):
                if unit["unit_name"] in ITEM_NAMES:
                    logger.debug(f"Returning {unit['unit_name']} because it has an item-like name")
                    return Option.some(_unit(unit, candidate))
                elif unit["unit_name"] in SKIP_NAMES:
                    logger.debug(f"Removing {unit['unit_name']} because it has a household measurement-like name")
                    rem.append(ix)
            rem.reverse()
            for r in rem:
                _=units.pop(r)
            units.sort(key=lambda it:it["seq_num"])
            unit = units[0]
            logger.debug(f"Returning {unit['unit_name']} because it has the lowest seq_num")
            return Option.some(_unit(unit, candidate))
    pstate = state.get(ParserState)
    if len(pstate.nonempty_amounts) > 0:
        logger.debug("Unit not empty, not looking up serving to avoid infinite recursion")
        return Option.none()
    logger.debug("No units, looking up `serving`")
    pstate_mod = ParserState(
        parsed_name=pstate.parsed_name,
        nonempty_amounts=[NonemptyAmount(
            confidence=1.0, quantity=1.0,unit="serving"
        )],
        empty_amount=Option.none(),
        outer_amount=Option.none(),
        prep=pstate.prep
    )
    state_msk = state.mask()
    state_msk.set(ParserState, pstate_mod)
    candi = state_msk.get_optional(UnitCandidate, allow_parent=False, from_provider=FDCUnitProvider)
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
        if len(pstate.nonempty_amounts) == 0:
            logger.debug("Can't provide fallback unit because unit is empty")
            return Option.none()
        for ix,amnt in enumerate(pstate.nonempty_amounts):
            unit =amnt.unit
            if unit in self.name_eq:
                res = Option.some_if(self.units.get(self.name_eq[unit]))
            else:
                res = Option.some_if(self.units.get(unit))
            if res.is_some():
                logger.debug(f"Found fallback for {unit} in database")
                res = res.map(lambda k:k.clone()).inspect(lambda k,ix=ix: k.parsed_from.set(ix))
                return res
            else:
                logger.debug(f"No fallback found for {unit}")
        return Option.none()
