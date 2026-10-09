import logging
from typing import override

from api.common import Option
from api.models import (
    Density,
    EmptyUnitResult,
    FallbackUnitCandidate,
    IngredientCandidate,
    ParserState,
)
from api.providers.unit_prediction import UnitRegressionPipeline, load_unit_model
from api.state import OptionalProvider, Provider, RequestState

REGRESSION_PENALTY = 0.2

class DensityRegressionProvider(OptionalProvider[Density]):
    pipeline:UnitRegressionPipeline

    def __init__(self, model_path:str, prompt_tmpl:str|None=None):
        super().__init__(Option[Density])
        self.pipeline = load_unit_model(model_path)
        self.pipeline.prompt="FOOD: {food}\nCOMMENTS: {comments}" if prompt_tmpl is None else prompt_tmpl
        self.pipeline.do_expm1=False

    @override
    def execute(self, state: RequestState) -> Option[Density]:
        logger = logging.getLogger("ingr_api").getChild("DensityRegr")
        cands = state.get_all(IngredientCandidate, only_use_available=True)
        pstate = state.get(ParserState)
        best_cand = max(cands, key=lambda k: k.score, default=None)
        if best_cand is not None and best_cand.score > 0.8:
            cand_name = best_cand.description
            score = best_cand.score
        else:
            logger.debug("No FDC candidates, using standard string")
            cand_name = pstate.parsed_name.unwrap_or("food")
            score = 0.5
        comments = pstate.prep.map(lambda k: [k]).unwrap_or([])
        dense = self.pipeline.predict_one(cand_name, None, comments)
        logger.debug(f"Inferred density as {dense}")
        return Option.some(Density(
            confidence=score * (1.0 - REGRESSION_PENALTY),
            density=dense
        ))

class NonstandardUnitRegrProvider(Provider[FallbackUnitCandidate]):
    pipeline: UnitRegressionPipeline

    def __init__(self, model_path:str, prompt_tmpl:str|None=None):
        super().__init__(FallbackUnitCandidate)
        self.pipeline = load_unit_model(model_path)
        self.pipeline.prompt="FOOD: {food}\nUNIT: {unit}\nCOMMENTS: {comments}" if prompt_tmpl is None else prompt_tmpl
        self.pipeline.use_unit_input=True

    @override
    def execute(self, state: RequestState) -> FallbackUnitCandidate:
        logger = logging.getLogger("ingr_api").getChild("UnitRegr")
        pstate = state.get(ParserState)
        cands = state.get_all(IngredientCandidate, only_use_available=True)
        best_cand = max(cands, key=lambda k: k.score, default=None)
        if best_cand is not None and best_cand.score > 0.8:
            cand_name = best_cand.description
        else:
            logger.debug("No FDC candidates, using standard string")
            cand_name = pstate.parsed_name.unwrap_or("food")
        logger.debug(f"Using {cand_name} as food")
        if len(pstate.nonempty_amounts) == 0:
            raise ValueError("Called NonstandardUnitRegrProvider when unit was empty/None!")
        from_ix,max_amnt = max(enumerate(pstate.nonempty_amounts), key=lambda k:k[1].confidence)
        unit_name = max_amnt.unit
        logger.debug(f"Using {unit_name} for inference")
        if unit_name[-1] == "s":
            singular_name = unit_name[:-1]
            plural_name = unit_name
        else:
            singular_name = unit_name
            plural_name = unit_name + "s"
        comments = pstate.prep.map(lambda k:[k]).unwrap_or([])
        wgt = self.pipeline.predict_one(cand_name, unit_name, comments)
        logger.debug(f"Got {wgt} as unit weight")
        cand=FallbackUnitCandidate(
            comments=comments,
            entry_id=-1,
            gram_weight=wgt,
            modifier=None,
            plural_name=plural_name,
            singular_name=singular_name,
            unit_name=unit_name,
            source="regression"
        )
        _=cand.parsed_from.set(from_ix)
        return cand

class EmptyUnitRegrProvider(Provider[EmptyUnitResult]):
    pipeline:UnitRegressionPipeline

    def __init__(self, model_path:str, prompt_tmpl:str|None=None):
        super().__init__(EmptyUnitResult)
        self.pipeline = load_unit_model(model_path)
        self.pipeline.prompt="FOOD: {food}\nCOMMENTS: {comments}" if prompt_tmpl is None else prompt_tmpl

    @override
    def execute(self, state: RequestState) -> EmptyUnitResult:
        logger = logging.getLogger("ingr_api").getChild("EmptyRegr")
        pstate = state.get(ParserState)
        cands = state.get_all(IngredientCandidate, only_use_available=True)
        best_cand = max(cands, key=lambda k: k.score, default=None)
        if best_cand is not None and best_cand.score > 0.8:
            cand_name = best_cand.description
        else:
            logger.debug("No FDC candidates, using standard string")
            cand_name = pstate.parsed_name.unwrap_or("food")
        logger.debug(f"Using {cand_name} as food")
        comments = pstate.prep.map(lambda k:[k]).unwrap_or([])
        wgt = self.pipeline.predict_one(cand_name, None, comments)
        logger.debug(f"Got {wgt} as weight")
        return EmptyUnitResult(
            confidence=0.8 * (1.0 - REGRESSION_PENALTY),
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
