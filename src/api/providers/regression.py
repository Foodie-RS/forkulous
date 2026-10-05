from typing import override

from typing_extensions import override

from api.common import Option
from api.models import Density, EmptyUnitResult, FallbackUnitCandidate, ParserState, UnitCandidate
from api.providers.fdc_ingredient import FDCCandidates
from api.providers.unit_prediction import UnitRegressionPipeline
from api.state import OptionalProvider, Provider, RequestState

REGRESSION_PENALTY = 0.2

class DensityRegressionProvider(OptionalProvider[Density]):
    pipeline:UnitRegressionPipeline

    def __init__(self):
        super().__init__(Option[Density])

    @override
    def execute(self, state: RequestState) -> Option[Density]:
        cands = state.get_optional(FDCCandidates)
        if cands.is_none_or(lambda k: len(k) == 0):
            return Option.none()
        cands_uw = cands.unwrap()
        pstate = state.get(ParserState)
        comments = pstate.prep.map(lambda k: [k]).unwrap_or([])
        _, best_cand = max(cands_uw.items(), key=lambda it: it[1].score)
        nam = best_cand.description
        dense = self.pipeline.predict_one(nam, None, comments)
        return Option.some(Density(
            confidence=best_cand.score * (1.0 - REGRESSION_PENALTY),
            density=dense
        ))

class NonstandardUnitRegrProvider(Provider[FallbackUnitCandidate]):
    pipeline: UnitRegressionPipeline

    @override
    def execute(self, state: RequestState) -> FallbackUnitCandidate:
        pstate = state.get(ParserState)
        cands = state.get_optional(FDCCandidates)
        if cands.is_none_or(lambda k:len(k) == 0):
            cand_name = pstate.parsed_name.unwrap_or("food")
        else:
            _, best_cand = max(cands.unwrap().items(), key=lambda k: k[1].score)
            cand_name = best_cand.description
        if pstate.unit.is_none_or(lambda k:k==""):
            raise ValueError("Called NonstandardUnitRegrProvider when unit was empty/None!")
        unit_name = pstate.unit.unwrap()
        if unit_name[-1] == "s":
            singular_name = unit_name[:-1]
            plural_name = unit_name
        else:
            singular_name = unit_name
            plural_name = unit_name + "s"
        comments = pstate.prep.map(lambda k:[k]).unwrap_or([])
        wgt = self.pipeline.predict_one(cand_name, unit_name, comments)
        return FallbackUnitCandidate(
            comments=comments,
            entry_id=-1,
            gram_weight=wgt,
            modifier=None,
            plural_name=plural_name,
            singular_name=singular_name,
            unit_name=unit_name,
            source="regression"
        )

class EmptyUnitRegrProvider(Provider[EmptyUnitResult]):
    pipeline:UnitRegressionPipeline

    def __init__(self):
        super().__init__(EmptyUnitResult)

    @override
    def execute(self, state: RequestState) -> EmptyUnitResult:
        pstate = state.get(ParserState)
        cands = state.get_optional(FDCCandidates)
        if cands.is_none_or(lambda k:len(k) == 0):
            cand_name = pstate.parsed_name.unwrap_or("food")
        else:
            _, best_cand = max(cands.unwrap().items(), key=lambda k: k[1].score)
            cand_name = best_cand.description
        comments = pstate.prep.map(lambda k:[k]).unwrap_or([])
        wgt = self.pipeline.predict_one(cand_name, None, comments)
        return EmptyUnitResult(
            confidence=0.5,
            res=FallbackUnitCandidate(
                comments=comments,
                entry_id=-1,
                gram_weight=wgt,
                modifier=None,
                plural_name="",
                singular_name="",
                source="regression",
                unit_name=""
            )
        )
