import json
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


@dataclass
class UnitRegressionPipeline:
    scaler:StandardScaler
    model:PreTrainedModel
    tokenizer:PreTrainedTokenizerBase
    use_unit_input:bool=True
    do_expm1:bool=True
    max_len:int=80
    prompt:str="FOOD: {food}\nUNIT: {unit}\nCOMMENTS: {comments}"

    def predict_one(self, food: str, unit:str|None, comments:list[str]|None):
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
            predictions = self.scaler.inverse_transform(outputs.logits.squeeze(-1).reshape(-1, 1)).squeeze(-1)
            if self.do_expm1:
                predictions = np.expm1(np.clip(predictions, a_min=None, a_max=20.0))
        return predictions

    def predict_many(self,food:list[str], unit:list[str]|None, comments:list[list[str]|None]) -> np.ndarray:
        if unit is None and self.use_unit_input:
            raise RuntimeError("Can't do inference without unit if use_unit_input is true")
        prompts:list[str] = []
        for i, fd in enumerate(food):
            comment = comments[i]
            if self.use_unit_input:
                un:str = unit[i]
                prompts.append(self.prompt.format(
                    food=fd,
                    unit=un,
                    comments="none" if comment is None else f"[{", ".join(comment)}]"
                ))
            else:
                prompts.append(self.prompt.format(
                    food=fd,
                    comments="none" if comment is None else f"[{", ".join(comment)}]"
                ))
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_len
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = self.scaler.inverse_transform(outputs.logits.squeeze(-1).reshape(-1, 1)).squeeze(-1)
            predictions = np.expm1(np.clip(predictions, a_min=None, a_max=20.0))
        return predictions

def load_unit_model(model_path:str, train_data_path:str) -> UnitRegressionPipeline:
    with open(train_data_path, "r") as f:
        train_dict = json.load(f)

    train_labels_raw = np.array(train_dict["label"], dtype=np.float64)

    #TODO scaler values in dedicated file
    scaler = StandardScaler()
    train_labels_log = np.log1p(train_labels_raw).reshape(-1, 1)
    scaler.fit(train_labels_log)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=1,
        problem_type="regression",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    model.eval()

    return UnitRegressionPipeline(
        model=model,
        scaler=scaler,
        tokenizer=tokenizer
    )
