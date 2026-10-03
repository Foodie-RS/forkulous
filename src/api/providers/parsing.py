import logging
from typing import override

import ingredient_parser
from ingredient_parser.dataclasses import (
    IngredientAmount,
    ParsedIngredient,
)

from api.common import Option
from api.models import (
    AmountModel,
    EmptyUnitResult,
    FallbackUnitCandidate,
    NameModel,
    ParseResults,
    ParserState,
    SearchRequest,
    SearchResult,
    ShortParseResult,
    UnitModel,
    S1_UnitNames_Res,
    S5_SelectUnits_Res,
)
from api.state import Provider, RequestState


class ParserProvider(Provider[ParsedIngredient]):
    @override
    def execute(self, state:RequestState) -> ParsedIngredient:
        logger = logging.getLogger("ingr_api").getChild("PrsrProv")
        search_req = state.get(SearchRequest)
        # Create combined unit_dict and unit_names
        unit_names_list:list[S1_UnitNames_Res] = state.multi_get(S1_UnitNames_Res)
        unit_names:dict[str, str] = {}
        for it in unit_names_list:
            unit_names.update(it.res)
        logger.debug(f"Got {len(unit_names)} unit names.")

        res = ingredient_parser.parse_ingredient(search_req.query, custom_units=unit_names, string_units=True)
        logger.debug(f"Parsed \"{search_req.query}\": {len(res.amount)} amounts, {len(res.name)} names, preparation: {"yes" if res.preparation is not None else "no"}")
        return res

class ParseResultProvider(Provider[ParseResults]):
    @override
    def execute(self, state:RequestState) -> ParseResults:
        logger = logging.getLogger("ingr_api").getChild("PrsResProv")
        logger.log(5, "Getting ParsedIngredient")
        parsed = state.get(ParsedIngredient)
        logger.log(5, "Got ParsedIngredient")

        # resolve amount and unit order
        outer_amount:float = 1
        if len(parsed.name) < 1:
            parsed_name = Option.none()
            logger.warning("Parsed name is empty")
        else:
            parsed_name:Option[str]= Option.some(parsed.name[0].text)
        logger.debug("Got %s amounts", len(parsed.amount))
        if len(parsed.amount) >= 1:
            all_amounts = parsed.amount
            if len(parsed.amount) > 1:
                has_singular = any(it.SINGULAR for it in parsed.amount)
                has_plural = any((not it.SINGULAR) for it in parsed.amount)
                if ((not has_singular) or (not has_plural) or (len(parsed.amount) > 2)):
                    logger.warning("Multiple amounts found")
                if has_singular and has_plural:
                    for it in parsed.amount:
                        assert isinstance(it, IngredientAmount)
                        if not it.SINGULAR:
                            logger.debug(f"Setting {it.text} as outer amount")
                            outer_amount = float(it.quantity)
                        else:
                            logger.debug(f"Setting {it.text} as inner amount")
                            all_amounts.append(it)
            prep:Option[str] = Option.none() if (parsed.preparation is None) or (len(parsed.preparation.text) == 0) else Option.some(parsed.preparation.text)
            best_selected_units = []
            best_score = 0
            best_amount = None
            best_inner_amount = 0.0
            best_parsed_unit:Option[str] = Option.none()
            for amnt in all_amounts:
                assert isinstance(amnt, IngredientAmount)
                if isinstance(amnt.quantity, str) and len(amnt.quantity) == 0:
                    logger.warning("Quantity is empty string, defaulting to 1.0")
                    actual_amount = 1.0
                else:
                    actual_amount = float(amnt.quantity)
                parsed_unit:Option[str]=Option.none() if ((not isinstance(amnt.unit, str)) or (amnt.unit == "")) else Option.some(amnt.unit)
                state_msk = state.mask()
                state_msk.set(ParserState, ParserState(
                    quantity=actual_amount,
                    unit=parsed_unit,
                    parsed_name=parsed_name,
                    prep=prep
                ))
                if (not amnt.unit) or (amnt.unit == ""):
                    actual_amount = float(amnt.quantity)
                    logger.debug("Empty unit, looking up default unit")
                    def_unit = state_msk.multi_get(EmptyUnitResult)
                    def_units = [k.res for k in def_unit]
                    def_units.sort(key=lambda k:k.relevance(), reverse=True)
                    new_unit = def_units[0]
                    score = new_unit.relevance()
                    if best_score < score:
                        best_score = score
                        best_selected_units = [(new_unit, score)]
                        best_amount = amnt
                        best_inner_amount = actual_amount
                        best_parsed_unit = Option.none()
                else:
                    # unit resolution
                    logger.debug("Unit Parsing: Looking up inner unit: %s", amnt.unit)
                    selected_units = state_msk.get(S5_SelectUnits_Res).res

                    logger.debug("Got units: [%s]", ", ".join([x[0].unit_name or "<null>" for x in selected_units]))
                    selected_units = selected_units[:3]
                    if len(selected_units) > 0 and best_score < selected_units[0][1]:
                        best_score = selected_units[0][1]
                        best_selected_units = selected_units
                        best_amount = amnt
                        best_inner_amount = 1 if amnt.quantity == "" else float(amnt.quantity)
                        best_parsed_unit = parsed_unit
                state_msk.invalidate()
        else: # if len(parsed.amount) == 0
            actual_amount = 1
            logger.debug("No amount found, defaulting to 1")
            def_unit = state.multi_get_first(FallbackUnitCandidate)
            best_selected_units = [(def_unit, None)]
            best_amount = None
            best_inner_amount = 1.0
            best_parsed_unit = Option.none()

        return ParseResults (
            name=[NameModel(
                text=it.text,
                confidence=it.confidence,
                starting_pos=it.starting_index
            ) for it in parsed.name],
            amount=AmountModel(
                quantity=best_inner_amount,
                parsed_unit=best_parsed_unit.to_union(),
                outer_amount=outer_amount,
                confidence=None if not best_amount else best_amount.confidence,
                resolved_units=[UnitModel(
                    name=it[0].unit_name or "",
                    comments=it[0].comments,
                    source=it[0].source,
                    relevance=it[1],
                    gram_weight=it[0].gram_weight
                ) for it in best_selected_units
                ]
            )
        )

class ShortParseResultProvider(Provider[ShortParseResult]):
    @override
    def execute(self, state:RequestState) -> ShortParseResult:
        parse_result = state.get(ParseResults)
        amnt = parse_result.amount
        qtty = amnt.quantity or 1
        assert len(amnt.resolved_units) > 0
        unit = amnt.resolved_units[0]
        unit_grams = unit.gram_weight
        total_grams = unit_grams * qtty * amnt.outer_amount
        ing_results = state.get(SearchResult)
        ing = ing_results.results[0]
        nutris = ing.nutris_100g
        final_nutris = nutris.multiply(total_grams / 100.0)

        parse_res = ShortParseResult(
            quantity=qtty,
            parsed_unit=amnt.parsed_unit,
            nutris_calculated=final_nutris,
            outer_amount=amnt.outer_amount,
            nutris_confidence=None if (unit.relevance is None or amnt.confidence is None) else (unit.relevance * ing.score * amnt.confidence),
            resolved_unit=unit
        )
        return parse_res
