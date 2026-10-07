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
    S2_UnitDict_Res,
)
from api.state import OptionalProvider, Provider, RequestState

PREPARATION_RELEVANCE_FACTOR=0.1
UNIT_EPSILON = 0.02
UNIT_RELEVANCE_CUTOFF = 0.7
INGREDIENT_HARD_CUTOFF = 0.1
UNIT_SEMANTIC_SCORE_RELEVANCE_FACTOR=0.4
UNIT_SCORE_CUTOFF=0.5
UNIT_UPPER_ACCEPT_LIMIT=0.8

def _rerank_prep(topk:list[UnitCandidate], unit_ranker:SentenceTransformer, state:RequestState) -> bool:
    logger = logging.getLogger("ingr_api").getChild("prep_search")
    amnt = state.get(ParserState)

    if amnt.prep.is_none():
        logger.warning("Called semantic prep search when prep was None")
        return False

    prep = amnt.prep.unwrap()

    # create prompt
    if amnt.parsed_name.is_some_and(lambda k: k!=""):
        prompt = f"means of preparing {amnt.parsed_name.unwrap()} in a recipe: "
    else:
        prompt = "means of preparing an ingredient in a recipe: "
    logger.debug(f"Using string: \"{prompt}{prep}\"")

    #TODO: replace unit_ranker with CE?
    query_embd = unit_ranker.encode(prep, prompt=prompt, show_progress_bar=False)
    # How similar is the query to just "{prompt}:''" ?
    base_simil = unit_ranker.similarity(unit_ranker.encode("", prompt=prompt, show_progress_bar=False), query_embd)

    max_prep = 0.0
    for it in topk:
        if len(it.comments) > 0:
            # only evaluate comments if it actually has comments
            score = 0
            for comment in it.comments:
                # try all comments to see if any are a preparation
                embd = unit_ranker.encode(comment, prompt=prompt, show_progress_bar=False)
                sim = float(unit_ranker.similarity(query_embd, embd))
                score = max(sim, score)
            max_prep = max(max_prep, score)
            score *= PREPARATION_RELEVANCE_FACTOR
            rel = it.relevance() * (1-PREPARATION_RELEVANCE_FACTOR) + score
            it.set_score(rel)
        else:
            # blanket penalty for nonspecific units
            rel = it.relevance() * (1-(PREPARATION_RELEVANCE_FACTOR / 2))
            it.set_score(rel)
    if (base_simil+UNIT_EPSILON) >= max_prep:
        logger.debug("Rejecting prep reranking: max_prep=%s<base_simil=%s", max_prep, base_simil)
        for k in topk:
            k.set_score(None)
        return False
    else:
        logger.debug("Accepting prep reranking: base_simil=%s<max_rel=%s", base_simil, max_prep)
        topk.sort(key=lambda k: k.relevance(), reverse=True)
        return True

class S5_SelectUnits_Prov(OptionalProvider[S5_SelectUnits_Res]):
    unit_ranker:SentenceTransformer
    @override
    def execute(self, state:RequestState) -> Option[S5_SelectUnits_Res]:
        #TODO: merge this provider with SemanticPrepSearchProvider?
        logger = logging.getLogger("ingr_api").getChild("UnitSel")
        logger.debug("Aggregating units")
        candidates:list[UnitCandidate] = []
        for cand in state.iter(UnitCandidate):
            candidates.append(cand)
            if cand.relevance() >= UNIT_UPPER_ACCEPT_LIMIT:
                # stop early if we get a good unit
                break

        logger.debug(f"Initial unit count: {len(candidates)} units")

        # dedup
        remove:list[int] = []
        candidate_ids = [it.entry_id for it in candidates]
        seen_names:set[tuple[str|None, str|None, str|None, str|None]] = set()
        for ix, it in enumerate(candidates):
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

        candidates.sort(key=lambda it:it.relevance(), reverse=True)
        # preparation as tiebreaker
        topk:list[UnitCandidate] = candidates[:max(len(candidates),10)]
        if len(topk) > 1 and parse_state.prep.is_some() and ((topk[0].relevance() - topk[1].relevance()) < UNIT_EPSILON):
            logger.debug("Ranking by preparation because multiple units qualify")
            _=_rerank_prep(state=state, topk=topk, unit_ranker=self.unit_ranker)

        #fallback
        if (len(topk) == 0 or topk[0].relevance() < UNIT_RELEVANCE_CUTOFF):
            logger.warning("Calling fallback providers because no suitable unit was found")
            un = state.get(FallbackUnitCandidate)
            rel = un.relevance()
            added = False
            for ix, cand in enumerate(topk):
                if cand.relevance() < rel:
                    topk.insert(ix, un)
                    added = True
                    break
            if not added:
                topk.append(un)

        logger.debug("Final unit count: %s", len(topk))

        return Option.some(S5_SelectUnits_Res(res=topk))
