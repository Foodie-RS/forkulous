from dataclasses import dataclass

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


@dataclass
class UnitRegressionPipeline:
    model:PreTrainedModel
    tokenizer:PreTrainedTokenizerBase
    use_unit_input:bool=False
    do_expm1:bool=True
    max_len:int=80
    prompt:str="FOOD: {food}\nUNIT: {unit}\nCOMMENTS: {comments}"

    def predict_one(self, food: str, unit:str|None, comments:list[str]|None) -> float:
        if unit is None and self.use_unit_input:
            raise RuntimeError("Can't do inference without unit if use_unit_input is true")
        if self.use_unit_input:
            prompt_fmtd = self.prompt.format(food=food,unit=unit,comments="none" if comments is None or len(comments) == 0 else f"[{", ".join(comments)}]")
        else:
            prompt_fmtd = self.prompt.format(food=food,comments="none" if comments is None or len(comments) == 0 else f"[{", ".join(comments)}]")
        inputs = self.tokenizer(
            prompt_fmtd,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_len
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs.logits.squeeze(-1)
            if self.do_expm1:
                predictions:np.ndarray = np.expm1(np.clip(predictions, a_min=None, a_max=20.0))
        return float(predictions)

def load_unit_model(model_path:str) -> UnitRegressionPipeline:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=1,
        problem_type="regression",
        torch_dtype=torch.float32,
        #low_cpu_mem_usage=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    model.eval()

    return UnitRegressionPipeline(
        model=model,
        tokenizer=tokenizer
    )
