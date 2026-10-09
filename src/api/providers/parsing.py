import logging
from dataclasses import dataclass
from typing import cast, override

import ingredient_parser
from ingredient_parser.dataclasses import (
    IngredientAmount,
    ParsedIngredient,
)

from api.common import Option, Result
from api.models import (
    AmountModel,
    NameModel,
    NonemptyAmount,
    ParseResults,
    ParserState,
    S1_UnitNames_Res,
    SearchRequest,
    ShortParseResult,
    UnitAggregateResult,
    UnitCandidate,
    UnitModel,
)
from api.state import Provider, RequestState

EQ_TO_EMPTY = ["medium", "normal", "regular"]
GOOD_UNIT_REL = 0.8

class ParserProvider(Provider[ParsedIngredient]):
    @override
    def execute(self, state:RequestState) -> ParsedIngredient:
        logger = logging.getLogger("ingr_api").getChild("PrsrProv")
        search_req = state.get(SearchRequest)
        # Create combined unit_dict and unit_names
        unit_names_list:list[S1_UnitNames_Res] = state.get_all(S1_UnitNames_Res)
        unit_names:dict[str, str] = {}
        for it in unit_names_list:
            unit_names.update(it.res)
        logger.debug(f"Got {len(unit_names)} unit names.")

        res = ingredient_parser.parse_ingredient(search_req.query, custom_units=unit_names, string_units=True)
        logger.debug(f"Parsed \"{search_req.query}\": {len(res.amount)} amounts, {len(res.name)} names, preparation: {"yes" if res.preparation is not None else "no"}")
        return res

@dataclass
class _AmountCandidate:
    unit:UnitCandidate
    qtty:float
    unit_parsed:Option[str]
    parser_conf:Option[float]

class ParserStateProvider(Provider[ParserState]):
    @override
    def execute(self, state:RequestState) -> ParserState:
        logger = logging.getLogger("ingr_api").getChild("ParserState")
        logger.log(5, "Getting ParsedIngredient")
        parsed = state.get(ParsedIngredient)
        logger.log(5, "Got ParsedIngredient")

        # resolve amount and unit order
        if len(parsed.name) < 1:
            parsed_name = Option.none()
            logger.warning("Parsed name is empty")
        else:
            parsed_name:Option[str]= Option.some(parsed.name[0].text)
        logger.debug("Got %s amounts", len(parsed.amount))
        amounts:list[IngredientAmount] = []
        outer_amount:Option[float] = Option.none()
        prep:Option[str] = Option.some_if(parsed.preparation).map(lambda k:k.text.strip()).filter(lambda k: len(k) > 0)
        amounts_pre = [cast(IngredientAmount, it) for it in parsed.amount]
        has_singular = any(it.SINGULAR for it in amounts_pre)
        has_plural = any((not it.SINGULAR) for it in amounts_pre)
        if ((not has_singular) or (not has_plural)) and (len(amounts_pre) > 2):
            logger.warning("Multiple distinct amounts found")
        if has_singular and has_plural:
            for amnt in amounts_pre:
                if not amnt.SINGULAR:
                    logger.debug(f"Setting {amnt.text} as outer amount")
                    outer_amount = Result.catch(ValueError, lambda: float(amnt.quantity)).ok_or_none().filter(lambda k: k>0)
                else:
                    logger.debug(f"Adding {amnt.text} as inner amount")
                    amounts.append(amnt)
        else:
            amounts = amounts_pre
        amounts_mapped:list[tuple[float, Option[str], float]] = []
        for amnt in amounts:
            unit = Option.some(cast(str, amnt.unit).strip().lower()).filter(lambda k: len(k) != 0).filter(lambda k: k not in EQ_TO_EMPTY)
            quantity = Result.catch(ValueError, lambda: float(amnt.quantity)).ok_or_none().filter(lambda k: k>0).unwrap_or(1.0)
            amounts_mapped.append((quantity, unit, amnt.confidence))
        empty_amnt = Option.some_if(max(((k[0], k[2]) for k in amounts_mapped if k[1].is_none()), key=lambda l:l[1], default=None))
        non_empty = [NonemptyAmount(quantity=k[0], unit=k[1].unwrap(), confidence=k[2]) for k in amounts_mapped if k[1].is_some()]
        return ParserState(
            nonempty_amounts=non_empty,
            empty_amount=empty_amnt,
            parsed_name=parsed_name,
            prep=prep,
            outer_amount=outer_amount
        )

class ParseResultProvider(Provider[ParseResults]):

    @override
    def execute(self, state: RequestState) -> ParseResults:
        logger = logging.getLogger("ingr_api").getChild("ParseResProvider")
        parsed_ingr = state.get(ParsedIngredient)
        pstate = state.get(ParserState)
        units = state.get(UnitAggregateResult).res
        models:list[AmountModel] = []
        for ix, amnt in enumerate(pstate.nonempty_amounts):
            # get all units for this nonempty amount
            unit_candidates = [k for k in units if k.parsed_from.filter(lambda i:i == ix).is_some()]
            if len(unit_candidates) == 0:
                logger.warning(f"No unit candidates for amount {amnt}")
                continue
            else:
                logger.debug(f"Found {len(unit_candidates)} units for {amnt}")
            unit_models = [
                UnitModel(
                    name=it.unit_name or "",
                    comments=it.comments,
                    source=it.source,
                    relevance=it.relevance(),
                    gram_weight=it.gram_weight,
                ) for it in unit_candidates
            ]
            models.append(AmountModel(
                confidence=amnt.confidence,
                outer_amount=pstate.outer_amount.unwrap_or(1.0),
                unit_parsed=amnt.unit,
                quantity=amnt.quantity,
                resolved_units=unit_models
            ))
        if pstate.empty_amount.is_some() or len(pstate.nonempty_amounts) == 0:
            quantity, confidence = pstate.empty_amount.unwrap_or_union((1.0,None))
            unit_candidates = [k for k in units if k.parsed_from.is_none_or(lambda k:k<0)]
            if len(unit_candidates) == 0:
                logger.warning("No unit candidates for amount empty amount")
            else:
                logger.debug(f"Found {len(unit_candidates)} units for empty amount")
                unit_models = [
                    UnitModel(
                        name=it.unit_name or "",
                        comments=it.comments,
                        source=it.source,
                        relevance=it.relevance(),
                        gram_weight=it.gram_weight,
                    ) for it in unit_candidates
                ]
                models.append(AmountModel(
                    confidence=confidence,
                    outer_amount=pstate.outer_amount.unwrap_or(1.0),
                    quantity=quantity,
                    unit_parsed=None,
                    resolved_units=unit_models
                ))

        return ParseResults (
            name=[NameModel(
                text=it.text,
                confidence=it.confidence,
                starting_pos=it.starting_index
            ) for it in parsed_ingr.name],
            amount=models
        )

class ShortParseResultProvider(Provider[ShortParseResult]):
    @override
    def execute(self, state:RequestState) -> ShortParseResult:
        parse_result = state.get(ParseResults)
        amnt = max(parse_result.amount, key=lambda k:k.confidence or 0.0)
        qtty = amnt.quantity or 1
        assert len(amnt.resolved_units) > 0
        unit = amnt.resolved_units[0]

        parse_res = ShortParseResult(
            quantity=qtty,
            outer_amount=amnt.outer_amount,
            resolved_unit=unit
        )
        return parse_res
