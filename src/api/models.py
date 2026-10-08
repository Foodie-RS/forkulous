from __future__ import annotations

import abc
import json
from dataclasses import dataclass
from typing import Any, NamedTuple, Self, override

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
    parsed_from:str|None

class AmountModel(BaseModel):
    quantity:float|None
    confidence:float|None
    outer_amount:float
    resolved_units:list[UnitModel]

class ParseResults(BaseModel):
    name:list[NameModel]
    amount:AmountModel

class ShortParseResult(BaseModel):
    quantity: float|None
    resolved_unit:UnitModel
    outer_amount:float
    nutris_calculated:Nutris
    nutris_confidence:float|None

class SearchParams(BaseModel):
    max_results_be:int=150
    max_results:int=3
    fast:bool=True
    parse:bool=True
    semantic_cutoff:float=0.1

    def arg_or_default(self,args:dict[str, str|bool]) -> SearchParams:
        return SearchParams(
            max_results_be=int(args.get("be_top_k", self.max_results_be)),
            max_results=int(args.get("ce_top_k", self.max_results)),
            fast=(args.get("fast") == "true" or args.get("fast") == True) if "fast" in args else self.fast,
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
class NutriSearchParam:
    cand:IngredientCandidate

@dataclass
class NutriSearchResult:
    result:Nutris

@dataclass
class S5_SelectUnits_Res:
    res:list[UnitCandidate]

@dataclass
class S6_SemanticPrep_Res:
    res:list[UnitCandidate]
@dataclass
class TransientUnitSelectionResult:
    res:list[UnitCandidate]
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
    unit:list[str]
    parsed_name:Option[str]
    prep:Option[str]


@dataclass
class UnitCandidate(metaclass=abc.ABCMeta):
    unit_name:str|None
    entry_id:int
    comments:list[str]
    singular_name:str|None
    plural_name:str|None
    gram_weight:float
    modifier:int|None
    source:str
    score_modifier:float = 1.0
    score_override:float|None=None
    parsed_from:str|None=None

    def relevance(self) -> float:
        if self.score_override is not None:
            return self.score_override
        return self.initial_relevance() * self.score_modifier

    @abc.abstractmethod
    def initial_relevance(self) -> float:
        return float("nan")

    def is_fallback(self) -> bool:
        return False

    def set_score(self, score:float|None):
        self.score_override = score

    @abc.abstractmethod
    def clone(self) -> UnitCandidate:
        pass

@dataclass
class S2_UnitDict_Res:
   res : dict[str,list[UnitCandidate]]

@dataclass
class S3_AggrDict_Res:
    res: dict[str, list[UnitCandidate]]

@dataclass
class S1_UnitNames_Res:
   res: dict[str, str]

class FallbackUnitCandidate(UnitCandidate):
    @override
    def initial_relevance(self) -> float:
        return 0.7

    @override
    def is_fallback(self) -> bool:
        return True

    @override
    def clone(self) -> FallbackUnitCandidate:
        return FallbackUnitCandidate(
            unit_name=self.unit_name,
            entry_id=self.entry_id,
            comments=self.comments,
            singular_name=self.singular_name,
            plural_name=self.plural_name,
            gram_weight=self.gram_weight,
            modifier=self.modifier,
            source=self.source,
            score_modifier=self.score_modifier
        )

@dataclass(init=False)
class PintUnitCandidate(UnitCandidate):
    density_confidence:float|None

    def __init__(self,
        unit_name:str|None,
        entry_id:int,
        comments:list[str],
        singular_name:str|None,
        plural_name:str|None,
        gram_weight:float,
        modifier:int|None,
        source:str,
        density_confidence:float|None,
        score_modifier:float = 1.0
    ):
        super().__init__(
            unit_name=unit_name,
            entry_id=entry_id,
            comments=comments,
            singular_name=singular_name,
            plural_name=plural_name,
            gram_weight=gram_weight,
            modifier=modifier,
            source=source,
            score_modifier=score_modifier
        )
        self.density_confidence = density_confidence

    @override
    def initial_relevance(self) -> float:
        return 0.8 + (self.density_confidence * 0.2) if self.density_confidence is not None else 0.8

    @override
    def clone(self) -> PintUnitCandidate:
        return PintUnitCandidate(
            unit_name=self.unit_name,
            entry_id=self.entry_id,
            comments=self.comments,
            singular_name=self.singular_name,
            plural_name=self.plural_name,
            gram_weight=self.gram_weight,
            modifier=self.modifier,
            source=self.source,
            score_modifier=self.score_modifier,
            density_confidence=self.density_confidence
        )


@dataclass(init=False)
class IngredientUnitCandidate(UnitCandidate):
    ingredient_candidate:IngredientCandidate
    measure_unit_id:int|None
    seq_num:int

    def __init__(self,
        unit_name:str|None,
        entry_id:int,
        comments:list[str],
        singular_name:str|None,
        plural_name:str|None,
        gram_weight:float,
        modifier:int|None,
        source:str,
        ingredient_candidate:IngredientCandidate,
        measure_unit_id:int|None,
        seq_num:int,
        score_modifier:float = 1.0
    ):
        super().__init__(
            unit_name=unit_name,
            entry_id=entry_id,
            comments=comments,
            singular_name=singular_name,
            plural_name=plural_name,
            gram_weight=gram_weight,
            modifier=modifier,
            source=source,
            score_modifier=score_modifier
        )
        self.ingredient_candidate = ingredient_candidate
        self.measure_unit_id=measure_unit_id
        self.seq_num=seq_num

    @override
    def initial_relevance(self) -> float:
        return self.ingredient_candidate.score * self.score_modifier

    @override
    def clone(self) -> IngredientUnitCandidate:
        return IngredientUnitCandidate(
            unit_name=self.unit_name,
            entry_id=self.entry_id,
            comments=self.comments,
            singular_name=self.singular_name,
            plural_name=self.plural_name,
            gram_weight=self.gram_weight,
            modifier=self.modifier,
            source=self.source,
            score_modifier=self.score_modifier,
            ingredient_candidate=self.ingredient_candidate,
            measure_unit_id=self.measure_unit_id,
            seq_num=self.seq_num
        )

class Density(NamedTuple):
    density:float
    confidence:float|None
