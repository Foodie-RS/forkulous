from __future__ import annotations

import logging
from typing import override

from api.models import (
    IngredientCandidate,
    IngredientModel,
    NutriSearchResult,
    SearchParams,
    SearchRequest,
    SearchResult,
)
from api.state import Provider, RequestState

GOOD_INGREDIENT_CUTOFF = 0.9


class IngredientSearchProvider(Provider[SearchResult]):
    @override
    def execute(self, state:RequestState) -> SearchResult:
        logger = logging.getLogger("ingr_api").getChild("IngrSrchProv")
        params:SearchParams = state.get(SearchRequest).search_params
        results:list[IngredientCandidate] = []
        for ingr in state.iter(IngredientCandidate):
            results.append(ingr)
            if params.fast and ingr.score > GOOD_INGREDIENT_CUTOFF:
                break
        results.sort(key=lambda it: it.score, reverse=True)
        logger.debug("Got %s results", len(results))
        results = results[:min(params.max_results, len(results))]

        results_new:list[IngredientModel] = []
        logger.debug("Searching nutris")
        for ing in results:
            msk = state.mask()
            msk.set(IngredientCandidate, ing)
            nutris = msk.get(NutriSearchResult).result
            results_new.append(IngredientModel(
                id=ing.id,
                description=ing.description,
                nutris_100g=nutris,
                score=ing.score
            ))
            msk.invalidate()
        return SearchResult(
            results=results_new
        )
