import logging
from dataclasses import dataclass
from itertools import chain
from typing import override

from sentence_transformers import SentenceTransformer

from api.common import Option
from api.models import (
    FallbackUnitCandidate,
    ParserState,
    S3_AggrDict_Res,
    S4_DictFilter_Res,
    S5_SelectUnits_Res,
    S6_SemanticPrep_Res,
    TransientUnitSelectionResult,
    UnitCandidate,
    UnitDictResult,
)
from api.state import OptionalProvider, Provider, RequestState

PREPARATION_RELEVANCE_FACTOR=0.1
UNIT_EPSILON = 0.02
UNIT_RELEVANCE_CUTOFF = 0.7
INGREDIENT_HARD_CUTOFF = 0.1
UNIT_SEMANTIC_SCORE_RELEVANCE_FACTOR=0.4
UNIT_SCORE_CUTOFF=0.5

@dataclass
class S6_SemanticPrep_Prov(Provider[S6_SemanticPrep_Res]):
    unit_ranker:SentenceTransformer
    @override
    def execute(self, state:RequestState) -> S6_SemanticPrep_Res:
        logger = logging.getLogger("ingr_api").getChild("prep_search")
        amnt = state.get(ParserState)
        topk = state.get(TransientUnitSelectionResult).res

        prep = amnt.prep
        if prep.is_none():
            logger.warning("Called semantic prep search when prep was None")
            return S6_SemanticPrep_Res(
                res=topk
            )

        candidates_new:list[tuple[UnitCandidate, float]] = []

        # create prompt
        if amnt.parsed_name.is_some_and(lambda k: k!=""):
            prompt = f"means of preparing {amnt.parsed_name.unwrap()} in a recipe: "
        else:
            prompt = "means of preparing an ingredient in a recipe: "
        logger.debug(f"Using string: \"{prompt}{prep}\"")

        query_embd = self.unit_ranker.encode(prep, prompt=prompt, show_progress_bar=False)
        # How similar is the query to just "{prompt}:''" ?
        base_simil = self.unit_ranker.similarity(self.unit_ranker.encode("", prompt=prompt, show_progress_bar=False), query_embd)

        max_prep = 0.0
        for (it, old_rel) in topk:
            if len(it.comments) > 0:
                # only evaluate comments if it actually has comments
                score = 0
                for comment in it.comments:
                    # try all comments to see if any are a preparation
                    embd = self.unit_ranker.encode(comment, prompt=prompt, show_progress_bar=False)
                    sim = float(self.unit_ranker.similarity(query_embd, embd))
                    score = max(sim, score)
                max_prep = max(max_prep, score)
                score *= PREPARATION_RELEVANCE_FACTOR
                rel = old_rel * (1-PREPARATION_RELEVANCE_FACTOR) + score
                candidates_new.append((it, old_rel * (1-PREPARATION_RELEVANCE_FACTOR) + score))
            else:
                # blanket penalty for nonspecific units
                rel = old_rel * (1-(PREPARATION_RELEVANCE_FACTOR / 2))
                candidates_new.append((it, rel))
        if (base_simil+UNIT_EPSILON) >= max_prep:
            logger.debug("Rejecting prep reranking: max_prep=%s<base_simil=%s", max_prep, base_simil)
            return S6_SemanticPrep_Res(res=topk)
        else:
            logger.debug("Accepting prep reranking: base_simil=%s<max_rel=%s", base_simil, max_prep)
            return S6_SemanticPrep_Res(res=candidates_new)

class S5_SelectUnits_Prov(OptionalProvider[S5_SelectUnits_Res]):
    @override
    def execute(self, state:RequestState) -> Option[S5_SelectUnits_Res]:
        #TODO: merge this provider with SemanticPrepSearchProvider?
        logger = logging.getLogger("ingr_api").getChild("UnitSel")
        logger.debug("Aggregating units")

        # get all units from all providers
        results:Option[S4_DictFilter_Res] = state.get_optional(S4_DictFilter_Res)
        if results.is_none_or(lambda k: len(k.res) == 0):
            logger.warning("DictFinder returned no results")
            return Option.none()
        candidates = results.unwrap().res
        logger.debug(f"Initial unit count: {len(candidates)} units")

        # dedup
        remove:list[int] = []
        candidate_ids = [it.entry_id for (it,_) in candidates]
        seen_names:set[tuple[str|None, str|None, str|None, str|None]] = set()
        for ix, (it, _) in enumerate(candidates):
            comment1 = None if len(it.comments) == 0 else it.comments[0]
            comment2 = None if len(it.comments) <=1 else it.comments[1]
            comment3 = None if len(it.comments) <=2 else it.comments[2]
            if it.entry_id in candidate_ids[:ix] or (it.unit_name, comment1, comment2, comment3) in seen_names:
                logger.log(5, f"Removing {it.unit_name} [{", ".join(it.comments)}] as duplicate")
                remove.append(ix)
            seen_names.add((it.unit_name, comment1, comment2, comment3))
        remove.reverse()
        for it in remove:
            _=candidates.pop(it)
        logger.debug(f"Count after deduplication: {len(candidates)}")

        parse_state = state.get(ParserState)

        # preparation as tiebreaker
        topk = candidates[:10]
        if len(topk) > 1 and parse_state.prep.is_some() and (((candidates[1][1]) - (topk[1][1])) < UNIT_EPSILON):
            logger.debug("Ranking by preparation because multiple units qualify")
            topk = state.get(S6_SemanticPrep_Res).res

        #sort
        topk.sort(key=lambda it:it[1], reverse=True)

        #fallback
        if (len(topk) == 0 or topk[0][1] < UNIT_RELEVANCE_CUTOFF):
            logger.warning("Calling fallback providers because no suitable unit was found")
            un = state.multi_get_first(FallbackUnitCandidate)
            rel = un.relevance()
            added = False
            for ix, (_, score) in enumerate(topk):
                if score < rel:
                    topk.insert(ix, (un, rel))
                    added = True
                    break
            if not added:
                topk.append((un, rel))

        logger.debug("Final unit count: %s", len(topk))

        return Option.some(S5_SelectUnits_Res(res=topk))


class S3_AggrDict_Prov(Provider[S3_AggrDict_Res]):
    @override
    def execute(self, state: RequestState) -> S3_AggrDict_Res:
        logger = logging.getLogger("ingr_api").getChild("AggrUnitDict")
        logger.debug("Aggregating results")
        all_dicts = state.multi_get(UnitDictResult, allow_deferred=True)
        logger.debug(f"Got {len(all_dicts)} result dicts, with a total of {sum(sum(len(it) for it in k.res.values()) for k in all_dicts)} entries")
        res:dict[str, list[UnitCandidate]] = {}
        for inner_res in all_dicts:
            for key,val in inner_res.res.items():
                if key in res:
                    res[key].extend(val)
                else:
                    res[key] = val
        logger.debug(f"Final dict has {len(res)} keys ({sum(len(k) for k in res.values())} entries)")
        return S3_AggrDict_Res(res=res)

def find_unit_in_dict (context:RequestState, unit:str, unit_ranker:SentenceTransformer) -> list[tuple[UnitCandidate, float]]:
    logger = logging.getLogger("ingr_api").getChild("FindUnitInDict")
    if unit == "":
        #return_candidates = [(it, it.relevance() * 0.7  + (0.3 * (5-_match_confidence_empty_unit(it))/4)) for it in dct[""]]
        raise ValueError("Called find_unit for an empty unit")
    logger.debug("Getting aggregate result")
    dct = context.get(S3_AggrDict_Res).res
    logger.debug(f"Got {len(dct)} keys ({sum(len(k) for k in dct.values())} entries)")
    return_keys:list[str] = []
    for k in dct:
        all_cand_names:set[str] = set()
        for cand in dct[k]:
            names = [it.lower() for it in [k, cand.singular_name, cand.plural_name] if it is not None]
            all_cand_names.update(names)
        if unit in all_cand_names:
            logger.debug(f"{k} added to return_keys")
            return_keys.append(k)
        else:
            logger.debug(f"{k} was not unit, so not adding to return_keys")
    return_candidates:list[tuple[UnitCandidate, float]] = [(it, it.relevance()) for it in chain(*[dct[k] for k in return_keys])]
    if len(return_candidates) == 0:
        prompt = ""
        candidate_keys = list(dct.keys())
        encoded_docs = unit_ranker.encode(candidate_keys, prompt=prompt)
        encoded_query = unit_ranker.encode(unit, prompt=prompt)
        for ix,key in enumerate(candidate_keys):
            if key == "":
                continue
            score = float(unit_ranker.similarity(encoded_docs[ix], encoded_query))
            candidates = dct[key]
            for cand in candidates:
                relevance =  (score * UNIT_SEMANTIC_SCORE_RELEVANCE_FACTOR) + (cand.relevance()) * (1-UNIT_SEMANTIC_SCORE_RELEVANCE_FACTOR)
                return_candidates.append((cand, relevance))
    remove_ix:list[int] = []
    for ix, (cand, score) in enumerate(return_candidates):
        if score < UNIT_SCORE_CUTOFF:
            logger.debug(f"{cand.unit_name}({cand.__class__.__name__}) was removed (score {score})")
            remove_ix.append(ix)
    logger.debug(f"Removing {len(remove_ix)} units, because their score was too low.")
    remove_ix.reverse()
    for it in remove_ix:
        _=return_candidates.pop(it)
    logger.debug(f"Returning {len(return_candidates)} items")
    return return_candidates

@dataclass
class S4_DictFilter_Prov(OptionalProvider[S4_DictFilter_Res]):
    unit_ranker:SentenceTransformer
    @override
    def execute(self, state:RequestState) -> Option[S4_DictFilter_Res]:
        logger = logging.getLogger("ingr_api").getChild("LclFDC_U_Prov")
        pstate = state.get_optional(ParserState)
        if pstate.is_none():
            logger.warning("No ParserState found!")
            return Option.none()
        unit = pstate.unwrap().unit
        if unit.is_none_or(lambda it: it == ""):
            logger.warning("Unit is None or empty")
            return Option.none()
        res= find_unit_in_dict(state, unit.unwrap(), self.unit_ranker)
        return Option.some(S4_DictFilter_Res(res))
