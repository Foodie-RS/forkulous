import json

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DebertaForSequenceClassification

MODEL_DIR="./models/unit_regression_6/checkpoint-6840"

with open("./fdc_unit_reg.train.json", "r") as f:
    train_dict = json.load(f)

train_labels_raw = np.array(train_dict["label"], dtype=np.float64)
assert np.all(np.isfinite(train_labels_raw)), "NaN oder Inf in train_dict['label'] gefunden!"
assert np.all(train_labels_raw >= 0), f"Negative Labels gefunden! Minimum: {train_labels_raw.min()}"

scaler = StandardScaler()
train_labels_log = np.log1p(train_labels_raw).reshape(-1, 1)
scaler.fit(train_labels_log)

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model:DebertaForSequenceClassification = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR,
    num_labels=1,
    problem_type="regression",
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
)
model.config.pad_token_id = tokenizer.pad_token_id

model.eval()

def add_scaler():
    with torch.no_grad():
        scale = float(scaler.scale_[0])
        mean = float(scaler.mean_[0])

        if hasattr(model.classifier, "out_proj"):
            final_layer = model.classifier.out_proj
        elif isinstance(model.classifier, nn.Linear):
            final_layer = model.classifier

        final_layer.weight.mul_(scale)
        final_layer.bias.mul_(scale).add_(mean)
    model.save_pretrained("./scaled_model")

def predict_weights(items: list[dict]) -> list[float]:
    prompts = []
    for item in items:
        food = item["food"]
        unit = item["unit"]
        comments = item.get("comments", "none")
        # Exakt das Format aus deinem Training:
        prompt = f"FOOD: {food}\nUNIT: {unit}\nCOMMENTS: {comments}"
        prompts.append(prompt)

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=80,
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = scaler.inverse_transform(outputs.logits.squeeze(-1).reshape(-1, 1)).squeeze(-1)
        predictions = np.expm1(np.clip(predictions, a_min=None, a_max=20.0))

    return predictions.tolist()


test_ingredients = [
    {"food": "Carrot, raw", "unit": "medium", "comments": "none"},
    {"food": "Ribs, NFS", "unit": "rack", "comments": "none"},
    {"food": "Celery, NFS", "unit": "stalk", "comments": "[diced]"},
]

#predicted_grams = predict_weights(test_ingredients)

#for item, grams in zip(test_ingredients, predicted_grams):
#    print(f"{item['unit']:>8} | {item['food']:<12} ({item['comments']}) -> {grams:6.1f} g")
add_scaler()
