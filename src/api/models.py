from __future__ import annotations

import abc
import json
from dataclasses import dataclass
from typing import Any, NamedTuple, override

from pydantic import BaseModel

from api.common import Option


class IngredientModel(BaseModel):
    id:str|int|None
    description:str
    score:float
    nutris_100g:Nutris

class NameModel(BaseModel):
    text:str
    starting_pos:int
    confidence:float

class UnitModel(BaseModel):
    name:str
    comments:list[str]
    source:str
    relevance:float|None
    gram_weight:float

class AmountModel(BaseModel):
    quantity:float|None
    parsed_unit:str|None
    confidence:float|None
    outer_amount:float
    resolved_units:list[UnitModel]

class ParseResults(BaseModel):
    name:list[NameModel]
    amount:AmountModel

class ShortParseResult(BaseModel):
    quantity: float|None
    parsed_unit: str|None
    resolved_unit:UnitModel
    outer_amount:float
    nutris_calculated:Nutris
    nutris_confidence:float|None

class SearchParams(BaseModel):
    max_results_be:int=150
    max_results:int=3
    exact:bool=True
    parse:bool=False
    semantic_cutoff:float=0.1

    def arg_or_default(self,args:dict[str, str|bool]) -> SearchParams:
        return SearchParams(
            max_results_be=int(args.get("be_top_k", self.max_results_be)),
            max_results=int(args.get("ce_top_k", self.max_results)),
            exact=(args.get("exact") == "true" or args.get("exact") == True) if "exact" in args else self.exact,
            parse=(args.get("parse") == "true" or args.get("parse") == True) if "parse" in args else self.parse,
        )

    def validate_params(self):
        if self.max_results_be < self.max_results:
            raise ValueError("TopK for transformer search cannot be smaller than TopK for CE search")

DEF_PARAMS = SearchParams()
class SearchRequest(BaseModel):
    query:str
    search_params:SearchParams=DEF_PARAMS

class SearchResponse(BaseModel):
    query:str
    search_results:list[IngredientModel]
    parse_result:ShortParseResult|None

class Nutris(BaseModel):
    energy:float|None=None
    carbs:float|None=None
    fat:float|None=None
    protein:float|None=None
    salt:float|None=None
    sat_fat:float|None=None
    sugar:float|None=None
    fiber:float|None=None
    calculated:list[str]|None=None

    def toJSON(self):
        return json.dumps(self.to_dict())

    def to_dict(self) -> dict[str, float|None|list[str]]:
        nutris:dict[str, Any]={k:v for k,v in [("energy", self.energy), ("carbs", self.carbs), ("fat", self.fat), ("protein", self.protein), ("salt", self.salt), ("sugar", self.sugar), ("fiber", self.fiber), ("satfat", self.sat_fat)]}
        if self.calculated is not None and len(self.calculated) > 0:
            nutris["calculated"] = self.calculated
        else:
            nutris["calculated"] = None
        return nutris

    def multiply(self, factor:float) -> Nutris:
        return Nutris (
            energy=None if self.energy is None else self.energy * factor,
            carbs=None if self.carbs is None else self.carbs * factor,
            fat=None if self.fat is None else self.fat * factor,
            protein=None if self.protein is None else self.protein * factor,
            salt=None if self.salt is None else self.salt * factor,
            sugar=None if self.sugar is None else self.sugar * factor,
            fiber=None if self.fiber is None else self.fiber * factor,
            sat_fat=None if self.sat_fat is None else self.sat_fat * factor,
            calculated=self.calculated
        )

@dataclass
class IngredientCandidate:
    id:str
    description:str
    score:float
    source:str

@dataclass
class SearchResult:
    results:list[IngredientModel]

@dataclass
class ProviderSearchResult:
    results:list[IngredientCandidate]

@dataclass
class NutriSearchResult:
    result:Nutris

@dataclass
class S5_SelectUnits_Res:
   res:list[tuple[UnitCandidate, float]]

@dataclass
class S6_SemanticPrep_Res:
   res:list[tuple[UnitCandidate, float]]
@dataclass
class TransientUnitSelectionResult:
   res:list[tuple[UnitCandidate, float]]
@dataclass
class S4_DictFilter_Res:
    res:list[tuple[UnitCandidate, float]]

@dataclass
class EmptyUnitResult:
    res:UnitCandidate
    confidence:float


@dataclass
class ParserState:
    quantity:float
    unit:Option[str]
    parsed_name:Option[str]
    prep:Option[str]


@dataclass(frozen=True)
class UnitCandidate(metaclass=abc.ABCMeta):
    unit_name:str|None
    entry_id:int
    comments:list[str]
    singular_name:str|None
    plural_name:str|None
    gram_weight:float
    modifier:int|None
    source:str

    @abc.abstractmethod
    def relevance(self) -> float:
        return float("nan")

    def is_fallback(self) -> bool:
        return False

@dataclass
class UnitDictResult:
   res : dict[str,list[UnitCandidate]]

@dataclass
class S3_AggrDict_Res:
    res: dict[str, list[UnitCandidate]]

@dataclass
class S1_UnitNames_Res:
   res: dict[str, str]

class FallbackUnitCandidate(UnitCandidate):
    @override
    def relevance(self) -> float:
        return 0.7

    @override
    def is_fallback(self) -> bool:
        return True

@dataclass(frozen=True)
class PintUnitCandidate(UnitCandidate):
    density_confidence:float|None
    @override
    def relevance(self) -> float:
        return 0.8 + (self.density_confidence * 0.2) if self.density_confidence is not None else 0.8

@dataclass(frozen=True)
class IngredientUnitCandidate(UnitCandidate):
    ingredient_candidate:IngredientCandidate
    measure_unit_id:int|None
    seq_num:int

    @override
    def relevance(self) -> float:
        return self.ingredient_candidate.score

class Density(NamedTuple):
    density:float
    confidence:float|None
