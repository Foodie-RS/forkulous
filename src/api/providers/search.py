from __future__ import annotations

import logging
from typing import Any, override

from api.models import (
    IngredientCandidate,
    IngredientModel,
    NutriSearchResult,
    ProviderSearchResult,
    SearchParams,
    SearchRequest,
    SearchResult,
)
from api.state import Provider, RequestState

INGREDIENT_HARD_CUTOFF = 0.1


class IngredientSearchProvider(Provider[SearchResult]):
    @override
    def execute(self, state:RequestState) -> SearchResult:
        logger = logging.getLogger("ingr_api").getChild("IngrSrchProv")
        params:SearchParams = state.get(SearchRequest).search_params
        results:list[IngredientCandidate] = []
        ingr_list = state.multi_get(ProviderSearchResult)
        for lst in ingr_list:
            results.extend(lst.results)

        results.sort(key=lambda it: it.score, reverse=True)
        logger.debug("Got %s results in total", len(results))
        results = results[:min(params.max_results, len(results))]

        results_new:list[IngredientModel] = []
        logger.debug("Searching nutris")
        for ing in results:
            msk = state.mask()
            msk.set(IngredientCandidate, ing)
            nutris = msk.multi_get_first(NutriSearchResult).result
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
