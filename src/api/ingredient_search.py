from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import pint
from ingredient_parser import UREG
from ingredient_parser.dataclasses import (
    ParsedIngredient,
)
from sentence_transformers import CrossEncoder, SentenceTransformer
from usearch.index import Index

from api.models import (
    EmptyUnitResult,
    FallbackUnitCandidate,
    IngredientCandidate,
    Nutris,
    NutriSearchResult,
    ParseResults,
    ParserState,
    S1_UnitNames_Res,
    SearchParams,
    SearchRequest,
    SearchResponse,
    SearchResult,
    ShortParseResult,
    UnitAggregateResult,
    UnitCandidate,
)
from api.providers.density import Density, FallbackDensityProvider, UnitDensityProvider
from api.providers.fdc_ingredient import (
    LocalFDCIngredientProvider,
    LocalFDCNutriProvider,
)
from api.providers.fdc_units import (
    FDCEmptyUnitProv,
    FDCFbUnitProv,
    FDCUnitNamesProv,
    FDCUnitProvider,
)
from api.providers.ingredient_search import (
    IngredientSearchProvider,
)
from api.providers.parsing import (
    ParseResultProvider,
    ParserProvider,
    ParserStateProvider,
    ShortParseResultProvider,
)
from api.providers.pint_unit import (
    PintUnitNamesProvider,
    PintUnitProvider,
)
from api.providers.regression import (
    DensityRegressionProvider,
    EmptyUnitRegrProvider,
    NonstandardUnitRegrProvider,
)
from api.providers.unit_aggregation import (
    S5_SelectUnits_Prov,
)
from api.state import (
    GeneratorProvider,
    OptionalProvider,
    Provider,
    RequestState,
    RootState,
)

BE_DEF_Q_PROMPT = "ingredient: "
BE_DEF_D_PROMPT = "usda: "

PREPARATION_RELEVANCE_FACTOR=0.1
UNIT_EPSILON = 0.02
UNIT_RELEVANCE_CUTOFF = 0.7

@dataclass(init=False)
class SearchAPI:
    def_search_params:SearchParams
    default_state:RootState

    def __init__(self, providers:dict[type[Any], list[Provider[Any]|GeneratorProvider[Any]]|Provider[Any]|GeneratorProvider[Any]], unit_ranker:SentenceTransformer, def_search_params:SearchParams):
        self.def_search_params = def_search_params
        state = RootState()
        state.add_provider(ParsedIngredient, ParserProvider(_type=ParsedIngredient))
        state.add_provider(ParserState, ParserStateProvider(_type=ParserState))
        state.add_provider(ParseResults, ParseResultProvider(_type=ParseResults))
        state.add_provider(ShortParseResult, ShortParseResultProvider(_type=ShortParseResult))
        state.add_provider(SearchResult, IngredientSearchProvider(_type=SearchResult))
        state.add_provider(UnitAggregateResult, S5_SelectUnits_Prov(_type=UnitAggregateResult))
        for cls, prov in providers.items():
            if isinstance(prov, list):
                for inner_prov in prov:
                    state.add_provider(cls, inner_prov)
            else:
                state.add_provider(cls, prov)
        self.default_state = state

    def add_provider[T](self, what:type[T], prov:Provider[T]|OptionalProvider[T], insert:bool=True):
        self.default_state.add_provider(what, prov, insert)

    def search_ingredient(self,req:SearchRequest)->SearchResponse:
        logger = logging.getLogger("ingr_api").getChild("search_ingredient")
        req.search_params.validate_params()
        root_state:RootState = self.default_state.clone()
        state = root_state.new_request()
        state.set(pint.UnitRegistry[Any], UREG)
        state.set(SearchRequest, req)
        if req.search_params.parse:
            parse_res = state.get(ShortParseResult)
            ingr_search_res = state.get(SearchResult)
            return SearchResponse(
                query=req.query,
                parse_result=parse_res,
                search_results=ingr_search_res.results
            )
        else:
            res = state.get(SearchResult)
            return SearchResponse(
                query=req.query,
                parse_result=None,
                search_results=res.results
            )


def _gen_index(corpus:dict[int,str], model:SentenceTransformer) -> Index:
    items:list[tuple[int,str]] = list(corpus.items())

    corpus_embd = model.encode_document([val for _, val in items], convert_to_numpy=True, batch_size=128, show_progress_bar=True)

    index = Index (
        ndim=corpus_embd.shape[1],
        metric="cos"
    )
    index.add([key for key,_ in items], corpus_embd)
    return index

def _load(be_model:str|SentenceTransformer,ce_model:str|CrossEncoder, unit_ranker:str|SentenceTransformer, corpus:dict[int,str], index_file,nutris_file, portions_file, kwargs) -> SearchAPI:
    logger = logging.getLogger("ingr_api")
    #FIXME clean up signature of load() methods
    model_be:SentenceTransformer
    if isinstance(be_model, SentenceTransformer):
        model_be = be_model
    else:
        model_be = SentenceTransformer(be_model)
    if isinstance(unit_ranker, SentenceTransformer):
        model_units = unit_ranker
    else:
        model_units = SentenceTransformer(unit_ranker)
    if os.path.exists(index_file):
        index = Index()
        index.load(index_file)
    else:
        logger.info("Regenerating index...")
        index = _gen_index(corpus, model_be)
        index.save(index_file)
    if isinstance(ce_model, CrossEncoder):
        model_ce = ce_model
    else:
        model_ce = CrossEncoder(ce_model, max_length=kwargs.get("max_sequence_len", 128))
    with open(nutris_file, "r") as f:
        nutris_corpus_ser = json.load(f)
    nutris_corpus = {}
    for k,v in nutris_corpus_ser.items():
        n = Nutris()
        [setattr(n, key, val) for key, val in v.items() if hasattr(n, key)]
        nutris_corpus[k]=n
    with open(portions_file, "r") as f:
        unit_corpus = json.load(f)
    def_search_params = SearchParams()
    ingr_prov = LocalFDCIngredientProvider(
        corpus=corpus,
        index=index,
        model_be=model_be,
        model_ce=model_ce
    )
    nutri_prov = LocalFDCNutriProvider(nutris_corpus)
    empty_prov = FDCEmptyUnitProv(
        unit_corpus=unit_corpus,
    )
    #TOOD replace CE with argument
    unit_prov = FDCUnitProvider(unit_corpus, unit_ranker=model_units)
    fdc_names = FDCUnitNamesProv(unit_corpus)
    pint_nam_prov = PintUnitNamesProvider(ureg=UREG)
    pint_prov = PintUnitProvider()
    fb_prov = FDCFbUnitProv(unit_corpus)
    dense_prov = UnitDensityProvider()
    fb_dense = FallbackDensityProvider()

    dense_regression = DensityRegressionProvider("Forkulous/tf-forkulous-dense-xs")
    empty_regression = EmptyUnitRegrProvider("Forkulous/tf-forkulous-empty-units-xs")
    unit_regression = NonstandardUnitRegrProvider("Forkulous/tf-forkulous-units-xs")


    providers:dict[type[Any], list[Provider[Any]|GeneratorProvider[Any]]|Provider[Any]|GeneratorProvider[Any]] = {
        IngredientCandidate:[ingr_prov],
        NutriSearchResult:[nutri_prov],
        EmptyUnitResult:[empty_prov, empty_regression],
        UnitCandidate:[pint_prov, unit_prov],
        S1_UnitNames_Res:[pint_nam_prov, fdc_names],
        FallbackUnitCandidate:[unit_regression, fb_prov],
        Density: [dense_prov, dense_regression, fb_dense]
    }

    return SearchAPI(providers=providers,def_search_params=def_search_params, unit_ranker=model_units)

def load2(be_model:str|SentenceTransformer,ce_model:str|CrossEncoder, unit_ranker:str|SentenceTransformer, index_file, nutri_file, portion_file,corpus_file_csv, id_column:str="id",def_search_params:SearchParams|None=None, **kwargs) -> SearchAPI:
    if not (os.path.isfile(corpus_file_csv) and os.path.exists(corpus_file_csv)):
        raise FileNotFoundError(corpus_file_csv)
    with open(corpus_file_csv, "r") as f:
        corpus = {}
        desc_col = kwargs.get("corpus_doc_column", "description")
        dr = csv.DictReader(f)
        for row in dr:
            corpus[int(row[id_column])] = row[desc_col]
    return _load(be_model, ce_model, unit_ranker, corpus, index_file, nutri_file, portion_file, kwargs=kwargs)
