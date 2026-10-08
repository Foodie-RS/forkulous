from __future__ import annotations
import math

from collections.abc import Generator
import logging
from dataclasses import dataclass
from typing import cast, override

import numpy as np
from mpmath import sigmoid
from sentence_transformers import CrossEncoder, SentenceTransformer
from usearch.index import Index

from api.common import Option
from api.models import (
    IngredientCandidate,
    Nutris,
    NutriSearchResult,
    ProviderSearchResult,
    SearchRequest,
)
from api.state import GeneratorProvider, OptionalProvider, Provider, RequestState

FDCCandidates = dict[str, IngredientCandidate]

FDC_CE_Q_PROMPT="query: "
FDC_CE_D_PROMPT="document: "
FDC_CE_ACTION_PROMPT=""
INGREDIENT_THRESH_RERANK=0.8

NutriProvider = OptionalProvider[NutriSearchResult]
IngredientProvider = GeneratorProvider[IngredientCandidate]
@dataclass
class LocalFDCIngredientProvider(IngredientProvider):
    model_be:SentenceTransformer
    model_ce:CrossEncoder
    index:Index
    corpus:dict[int,str]

    def __init__(self,
        model_be:SentenceTransformer,
        model_ce:CrossEncoder,
        index:Index,
        corpus:dict[int,str]
    ):
        super().__init__(_type=IngredientCandidate)
        self.model_be = model_be
        self.model_ce = model_ce
        self.index = index
        self.corpus = corpus

    @override
    def execute(self, state:RequestState) -> Generator[IngredientCandidate, None, None]:
        logger = logging.getLogger("ingr_api").getChild("local_fdc").getChild("search_ingr")
        req = state.get(SearchRequest)
        ingr = req.query
        embd:np.ndarray = cast(np.ndarray, self.model_be.encode_query(ingr, convert_to_numpy=True))
        result_be_1:list[tuple[int, str, float]] = [(match.key, self.corpus[match.key], 1.0-match.distance) for match in self.index.search(embd, count=req.search_params.max_results_be, exact=not req.search_params.fast)]
        docs:list[str] = []
        results_be:list[tuple[int, str, float]] = []
        for id, desc, score in result_be_1:
            if desc in docs:
                continue
            docs.append(desc)
            results_be.append((id, desc, score))
        above_thresh_encountered = False
        count_below_thresh = 0
        ix = 0
        for ix,res in enumerate(results_be):
            if not above_thresh_encountered and ix >= 3:
                break
            if count_below_thresh > 7:
                logger.debug("Good ingredient found, but encountered too many consecutive bad ingredients. Bailing out.")
                return
            score = float(self.model_ce.predict((f"{FDC_CE_Q_PROMPT}{ingr}", f"{FDC_CE_D_PROMPT}{res[1]}"), prompt=FDC_CE_ACTION_PROMPT))
            score_sigm = 1 / (1 + math.exp(-score))
            if score_sigm > INGREDIENT_THRESH_RERANK:
                above_thresh_encountered = True
                count_below_thresh = 0
            else:
                count_below_thresh += 1
            logger.debug(f"Yielding {res[1]} (score {score_sigm})")
            yield IngredientCandidate(
                id=str(res[0]),
                description=res[1],
                score=score_sigm,
                source="fdc"
            )
        if above_thresh_encountered:
            return
        logger.debug("No clear winner found, reranking all")
        reranked_list = self.model_ce.rank(f"{FDC_CE_Q_PROMPT}{ingr}", [f"{FDC_CE_D_PROMPT}{res[1]}" for res in results_be[ix:]], prompt=FDC_CE_ACTION_PROMPT, top_k=10)
        new_results:list[tuple[int, str, float]] = [(results_be[it["corpus_id"]][0], results_be[it["corpus_id"]][1], (1/(1+math.exp(-it["score"])))) for it in reranked_list]
        for id,desc,score in new_results:
            if score < 0.2:
                # TODO: make this a static variable
                return
            logger.debug(f"Yielding {desc} (score {score})")
            yield IngredientCandidate(
                id=str(id),
                description=desc,
                score=score,
                source="fdc"
            )

@dataclass
class LocalFDCNutriProvider(NutriProvider):
    nutris:dict[str, Nutris]

    def __init__(self, nutris:dict[str, Nutris]):
        super().__init__(Option[NutriSearchResult])
        self.nutris = nutris

    @override
    def execute(self, state: RequestState) -> Option[NutriSearchResult]:
        cand = state.get(IngredientCandidate)
        if cand.source != "fdc":
            return Option.none()
        id = int(cand.id)
        return Option.some(NutriSearchResult(result=self.nutris[str(id)]))
