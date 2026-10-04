from __future__ import annotations

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
from api.state import OptionalProvider, Provider, RequestState

FDCCandidates = dict[str, IngredientCandidate]

FDC_CE_Q_PROMPT="query: "
FDC_CE_D_PROMPT="document: "
FDC_CE_ACTION_PROMPT=""

NutriProvider = OptionalProvider[NutriSearchResult]
IngredientProvider = Provider[ProviderSearchResult]
@dataclass
class LocalFDCIngredientProvider(IngredientProvider):
    model_be:SentenceTransformer
    model_ce:CrossEncoder
    index:Index
    corpus:dict[int,str]
    @override
    def execute(self, state:RequestState) -> ProviderSearchResult:
        logger = logging.getLogger("ingr_api").getChild("local_fdc").getChild("search_ingr")
        req = state.get(SearchRequest)
        ingr = req.query
        embd:np.ndarray = cast(np.ndarray, self.model_be.encode_query(ingr, convert_to_numpy=True))
        result_be:list[tuple[int, str]] = [(match.key, self.corpus[match.key]) for match in self.index.search(embd, count=req.search_params.max_results_be, exact=req.search_params.exact)]
        docs:list[str] = []
        new_result:list[tuple[int, str]] = []
        for id, desc in result_be:
            if desc in docs:
                continue
            docs.append(desc)
            new_result.append((id, desc))
        result_be = new_result
        result_ce = self.model_ce.rank(f"{FDC_CE_Q_PROMPT}{ingr}", [f"{FDC_CE_D_PROMPT}{nam}" for _,nam in result_be], top_k=req.search_params.max_results_be, prompt=FDC_CE_ACTION_PROMPT)

        result_list = [IngredientCandidate(
            id= str(result_be[res["corpus_id"]][0]),
            description= result_be[res["corpus_id"]][1],
            score=(float(sigmoid(res["score"]))**5),
            source="fdc"
        )
         for res in result_ce]
        final_cands = [res for ix, res in enumerate(result_list) if ix == 0 or res.score > req.search_params.semantic_cutoff]
        logger.debug("Removed %s candidates because their score was too low", len(result_list) - len(final_cands))
        #TODO find a way to tell RequestState that FDCCandidates will be set by this function (maybe make FDCCandidates a subclass of ProviderSearchResult?)
        state.set(FDCCandidates, {it.id:it for it in final_cands})
        return ProviderSearchResult(
            results=final_cands
        )

@dataclass
class LocalFDCNutriProvider(NutriProvider):
    nutris:dict[str, Nutris]

    @override
    def execute(self, state: RequestState) -> Option[NutriSearchResult]:
        cand = state.get(IngredientCandidate)
        if cand.source != "fdc":
            return Option.none()
        id = int(cand.id)
        return Option.some(NutriSearchResult(result=self.nutris[str(id)]))
