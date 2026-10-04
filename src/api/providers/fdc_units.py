import logging
from dataclasses import dataclass
from typing import Any, cast, override

from sentence_transformers import SentenceTransformer

from api.common import Option
from api.models import (
    EmptyUnitResult,
    FallbackUnitCandidate,
    IngredientUnitCandidate,
    ParserState,
    S1_UnitNames_Res,
    UnitCandidate,
    UnitDictResult,
)
from api.providers.fdc_ingredient import FDCCandidates
from api.providers import unit_aggregation
from api.state import OptionalProvider, Provider, RequestState

SERVING_MEASURE_UNITS = [1036,1049,1059,1069,1071,1096]
ITEM_MEASURE_UNITS = list(range(1013,1030)) + list(range(1031, 1038)) + list(range(1039,1049)) + list(range(1050,1059)) + [1060] + list(range(1063,1069)) + [1070] + list(range(1072,1121))
ITEM_NAMES=["fruit", "piece", ""]

def _cmp(items:list[tuple[Any, Any, bool]]):
    """
      Convenience method to compare multiple attributes of two arbitrary objects.
      Evaluates attributes in specified order, returning on the first attribute that is not equal.
    """
    for a,b, greater in items:
        if a and b:
            if (greater and (a > b)) or ((not greater) and (a < b)):
                return True
            elif (greater and (b > a)) or ((not greater) and (b < a)):
                return False
    return None

@dataclass
class FDCUnitNamesProv(Provider[S1_UnitNames_Res]):
    unit_corpus:dict[str, list[dict[Any, Any]]]
    @override
    def execute(self, state: RequestState) -> S1_UnitNames_Res:
        names:dict[str, str] = {}
        fdc_candidates = state.get(FDCCandidates)
        for cand in fdc_candidates.values():
            units = self.unit_corpus[cand.id]
            for unit in units:
                if unit["unit_name"] != "":
                    names[unit["plural_name"]] = unit["singular_name"]
        return S1_UnitNames_Res(res=names)

@dataclass
class LocalFDCUnitDictProv(Provider[UnitDictResult]):
    unit_corpus:dict[str, list[dict[Any, Any]]]
    @override
    def execute(self, state: RequestState) -> UnitDictResult:
        logger = logging.getLogger("ingr_api").getChild("FDCUnitProv")
        fdc_candidates = state.get(FDCCandidates)
        logger.debug(f"Got {len(fdc_candidates)} candidates")
        #TODO: replace generator with for loop
        def gener():
            for id,candidate in fdc_candidates.items():
                cand_units = self.unit_corpus[str(id)]
                for unit in cand_units:
                    new_unit = IngredientUnitCandidate(
                        measure_unit_id=unit["measure_unit_id"],
                        ingredient_candidate=candidate,
                        comments=unit["comments"],
                        modifier=unit["modifier"],
                        entry_id=unit["entry_id"],
                        gram_weight=unit["gram_weight"],
                        plural_name=unit["plural_name"],
                        singular_name=unit["singular_name"],
                        seq_num=unit["seq_num"],
                        unit_name=unit["unit_name"],
                        source=unit["source"]
                    )
                    yield new_unit


        units: dict[str, list[UnitCandidate]] = {}
        for new_unit in gener():
            added = False
            if new_unit.unit_name is not None and len(new_unit.unit_name) > 0:
                if new_unit.unit_name in units:
                    old_units = units[new_unit.unit_name]
                    added = False
                    for ix, unit in enumerate(old_units):
                        old_better = _cmp([
                            (unit.relevance(), new_unit.relevance(), True),
                            (cast(IngredientUnitCandidate, unit).seq_num, new_unit.seq_num, False),
                            (len(unit.comments), len(new_unit.comments), False),
                        ])
                        if not old_better:
                            old_units.insert(ix, new_unit)
                            added = True
                            break
                    if not added:
                        old_units.append(new_unit)
                else:
                    units[new_unit.unit_name] = [new_unit]
        logger.debug(f"Got a total of {len(units)} units.")
        return UnitDictResult(res=units)

def _get_default(unit_ranker:SentenceTransformer, crps:dict[str, list[dict[Any, Any]]], state:RequestState) -> Option[UnitCandidate]:
    logger = logging.getLogger("ingr_api").getChild("_get_def_un")
    cand = state.get(FDCCandidates)
    cnd_sorted = list(cand.values())
    cnd_sorted.sort(key=lambda k: k.score, reverse=True)
    logger.debug(f"Got {len(cnd_sorted)} candidates")
    for candidate in cnd_sorted:
        #TODO make this a static const variable
        if candidate.score < 0.2:
            continue
        logger.debug(f"Looking up {candidate.description}")
        if str(candidate.id) in crps:
            units = list(crps[str(candidate.id)])
            if len(units) == 0:
                continue
            units.sort(key=lambda it:it["seq_num"])
            unit = units[0]
            logger.debug(f"Returning {unit['unit_name']} because it has the lowest seq_num")
            return Option.some(IngredientUnitCandidate(
                measure_unit_id=unit["measure_unit_id"],
                ingredient_candidate=candidate,
                comments=unit["comments"],
                modifier=unit["modifier"],
                entry_id=unit["entry_id"],
                gram_weight=unit["gram_weight"],
                plural_name=unit["plural_name"],
                singular_name=unit["singular_name"],
                seq_num=unit["seq_num"],
                unit_name=unit["unit_name"],
                source=unit["source"]
            ))
    logger.debug("No units, looking up `serving`")
    #TODO maybe generalize this more? "serving" may be too narrow?
    res = unit_aggregation.find_unit_in_dict(state, "serving", unit_ranker)
    if len(res) == 0:
        logger.debug("No `serving` found.")
        return Option.none()
    elif res[0][1] < 0.2:
        logger.debug("Score too low, returning None")
        return Option.none()
    else:
        return Option.some(res[0][0])

@dataclass
class FDCEmptyUnitProv(OptionalProvider[EmptyUnitResult]):
    unit_corpus:dict[str, list[dict[Any, Any]]]
    unit_ranker:SentenceTransformer

    @override
    def execute(self, state:RequestState) -> Option[EmptyUnitResult]:
        res = _get_default(self.unit_ranker, self.unit_corpus, state)
        return res.map(lambda res:EmptyUnitResult(res=res, confidence=res.relevance()))


@dataclass
class FDCFbUnitProv(OptionalProvider[FallbackUnitCandidate]):
    unit_corpus:dict[str, list[dict[Any, Any]]]
    @override
    def execute(self, state:RequestState) -> Option[FallbackUnitCandidate]:
        logger = logging.getLogger("ingr_api").getChild("FDCFbProv")
        pstate = state.get(ParserState)
        if pstate.unit.is_none_or(lambda k:k == ""):
            logger.debug("Can't provide fallback unit because unit is empty")
            return Option.none()
        unit = pstate.unit.unwrap()
        sum_gr:float = 0
        cnt_gr:float = 0
        name_low = unit.lower()
        if unit.endswith("s"):
            name_sngl = name_low[:-1]
            name_plrl = name_low
        else:
            name_sngl = name_low
            name_plrl = name_low + "s"
        for v in self.unit_corpus.values():
            for it_unit in v:
                if not it_unit["unit_name"]:
                    continue
                if it_unit["unit_name"].lower() == name_sngl or it_unit["unit_name"].lower() == name_plrl:
                    sum_gr += it_unit["gram_weight"]
                    cnt_gr += 1
        if cnt_gr == 0:
            return Option.none()
        gr_wgt = sum_gr / cnt_gr
        cand = FallbackUnitCandidate(
            comments=[],
            entry_id=-1,
            gram_weight=gr_wgt,
            singular_name=name_sngl,
            plural_name=name_plrl,
            modifier=None,
            source="f_fallback",
            unit_name=unit
        )
        return Option.some(cand)

#@dataclass
#class FDCInferenceUnitProvider(FallbackUnitProvider):
#    pipeline_nonstandard:UnitRegressionPipeline
#    pipeline_empty:UnitRegressionPipeline
#    @override
#    def find_fallback_unit(self, unit: str, comments:list[str], ingredient_parsed:str|None, context:RequestState) -> UnitCandidate | None:
#        logger = logging.getLogger("ingr_api").getChild("unit_inf_find_fb")
#        logger.debug(f"Find unit for \"{unit}\" of \"{ingredient_parsed}\".")
#        candidates= context.get(CandidatesType)
#        cand_list = list(candidates.values())
#        logger.debug(f"Ingredients in candidate list: [{", ".join([k.description for k in cand_list])}]")
#        scores = np.array([k.score for k in cand_list])
#        if unit.endswith("s"):
#            sngl_name = unit[:-1]
#            plrl_name = unit
#        else:
#            sngl_name = unit
#            plrl_name = unit + "s"
#
#        logger.debug(f"Scores: {scores}")
#        if len(cand_list) == 0:
#            if ingredient_parsed is not None:
#                logger.debug("No candidates found. Defaulting to parsed ingredient.")
#            else:
#                logger.debug("No candidates found and parsed ingredient is None. Defaulting to \"Food Item, NFS\"")
#                ingredient_parsed = "Food Item, NFS"
#
#            grams = self.pipeline_nonstandard.predict_one(ingredient_parsed, unit, comments)
#            logger.debug(f"Inference finished. Reported {grams} grams.")
#            return FallbackUnitCandidate(
#                comments=comments,
#                unit_name=unit,
#                plural_name=plrl_name,
#                singular_name=sngl_name,
#                gram_weight=float(grams),
#                entry_id=-1,
#                modifier=None,
#                source="f_fallback"
#            )
#        unit_list = [unit] * len(cand_list)
#        comment_list:list[list[str]|None] = [comments] * len(cand_list)
#        grams = self.pipeline_nonstandard.predict_many([k.description for k in cand_list], unit_list, comment_list)
#        logger.debug(f"Got grams: {grams}")
#        total_score = np.sum(scores)
#        mean_grams = np.sum(grams * scores) / total_score
#        logger.debug(f"Calculated weighted mean grams as {mean_grams}")
#
#        return FallbackUnitCandidate(
#            entry_id=-1,
#            comments=comments,
#            unit_name=unit,
#            gram_weight=float(mean_grams),
#            modifier=None,
#            plural_name=plrl_name,
#            singular_name=sngl_name,
#            source="f_fallback"
#        )
