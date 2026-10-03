import json

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from scipy.stats import pearsonr
from peft import LoraConfig, get_peft_model, TaskType

DO_LORA=False
ADD_CLASS_LAYER=False
MODEL_NAME="microsoft/deberta-v3-large"
IS_LLM=False
STANDARD_SCALER=True
L1_LOSS=False
DO_LOG1P=False
DO_EXP=False
#MODEL_NAME="Qwen/Qwen2.5-7B"
#IS_LLM=True
MAX_LENGTH = 60
WARMUP_RT=0.1
NUM_EPOCHS=4

#=== Hyperparameters ===
LR=5e-5
LR_HEAD=1e-4
ACTUAL_BATCH=16
GR_ACC=1

#=== LORA ====
LORA_TRAIN_EMBEDDINGS=True
LR_LORA=5e-5
ACTUAL_BATCH_LORA=2
GR_ACC_LORA=8

EVALS_PER_EPOCH=3

loss_fn = nn.SmoothL1Loss(beta=0.25)

def smooth_l1_loss_func(outputs, labels, num_items_in_batch=None):
    logits = outputs.logits.squeeze(-1)
    return loss_fn(logits, labels)

LORA=DO_LORA or IS_LLM
with open("./fdc_unit_reg_dense.eval.json", "r") as f:
    eval_dict = json.load(f)

with open("./fdc_unit_reg_dense.train.json", "r") as f:
    train_dict = json.load(f)

if IS_LLM:
    train_dict["text"] = [k + "\nWEIGHT: " for k in train_dict["text"]]
    eval_dict["text"] = [k + "\nWEIGHT: " for k in eval_dict["text"]]

GR_ACC_ACT = GR_ACC_LORA if LORA else GR_ACC
A_BATCH_ACT = ACTUAL_BATCH_LORA if LORA else ACTUAL_BATCH

STEPS_PER_EPOCH = len(train_dict["text"]) / (GR_ACC_ACT * A_BATCH_ACT)

WARMUP_STEPS = int(NUM_EPOCHS * WARMUP_RT * STEPS_PER_EPOCH)

train_labels_raw = np.array(train_dict["label"], dtype=np.float64)
eval_labels_raw = np.array(eval_dict["label"], dtype=np.float64)
assert np.all(np.isfinite(train_labels_raw)), "NaN oder Inf in train_dict['label'] gefunden!"
assert np.all(train_labels_raw >= 0), f"Negative Labels gefunden! Minimum: {train_labels_raw.min()}"

scaler = StandardScaler()
if DO_LOG1P:
    train_labels_log = np.log1p(train_labels_raw).reshape(-1, 1)
elif DO_EXP:
    train_labels_log = np.expm1(train_labels_raw).reshape(-1,1)
else:
    train_labels_log = train_labels_raw.reshape(-1, 1)
scaler.fit(train_labels_log)
ds_eval = Dataset.from_dict(eval_dict)
ds_train = Dataset.from_dict(train_dict).shuffle(42)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def preprocess_function(examples):
    tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )
    if DO_LOG1P:
        values = np.log1p(examples["label"])
    elif DO_EXP:
        values = np.expm1(examples["label"])
    else:
        values = np.array(examples["label"])
    if STANDARD_SCALER:
        values = scaler.transform(values.reshape(-1, 1)).squeeze(-1)

    tokenized["label"] = values.astype(np.float32).tolist()
    return tokenized

tok_ds_eval = ds_eval.map(preprocess_function, batched=True)
tok_ds_train = ds_train.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=1,
    problem_type="regression",
    torch_dtype=torch.float32,
)
if model.config.pad_token_id is None:
    model.config.pad_token_id = tokenizer.pad_token_id

mean_log_target = float(np.mean(tok_ds_train["label"]))
if ADD_CLASS_LAYER and not IS_LLM:
    hidden_size = model.config.hidden_size
    model.classifier = nn.Sequential(
        nn.Linear(hidden_size, 256),
        nn.LayerNorm(256),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(256, 1)
    )
    # Bias der letzten Schicht auf den Log-Mittelwert setzen:
    with torch.no_grad():
        model.classifier[-1].bias.fill_(mean_log_target)
elif IS_LLM:
    hidden_size = model.config.hidden_size
    model.score = nn.Linear(hidden_size, 1, bias=True).to(model.dtype)

    with torch.no_grad():
        nn.init.normal_(model.score.weight, mean=0.0, std=0.01)
        model.score.bias.fill_(mean_log_target)
else:
    with torch.no_grad():
        if hasattr(model.classifier, "out_proj"):  # RoBERTa
            #nn.init.normal_(model.classifier.out_proj.weight, mean=0.0, std=0.001)
            model.classifier.out_proj.bias.fill_(mean_log_target)
        elif isinstance(model.classifier, nn.Linear):  # DeBERTa / DistilBERT
            #nn.init.normal_(model.classifier.weight, mean=0.0, std=0.001)
            model.classifier.bias.fill_(mean_log_target)

if LORA:
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        target_modules=["word_embeddings", "query_proj", "key_proj", "value_proj", "dense"] if LORA_TRAIN_EMBEDDINGS and not IS_LLM else "all-linear",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if STANDARD_SCALER:
        predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).squeeze(-1)
        labels = scaler.inverse_transform(labels.reshape(-1, 1)).squeeze(-1)
    else:
        predictions = predictions.squeeze(-1)

    predictions = np.clip(predictions, a_min=None, a_max=20.0)
    if DO_LOG1P:
        predictions = np.expm1(predictions)
        labels = np.expm1(labels)
    elif DO_EXP:
        predictions = np.log1p(predictions)
        labels = np.log1p(labels)
    abs_err = np.abs(labels - predictions)
    mae = np.mean(abs_err)
    rmse = np.sqrt(np.mean(abs_err ** 2))
    medae = np.median(abs_err)  # 50% aller Schätzungen sind genauer als dieser Wert!
    mape = np.mean(abs_err / np.maximum(labels, 1.0)) * 100  # Prozentualer Fehler
    stdev = np.std(abs_err / np.maximum(labels, 1.0)) * 100
    pearson_corr, _ = pearsonr(labels, predictions)

    return {
        "mae_g": mae,
        "rmse_g": rmse,
        "median_err_g": medae,
        "mape_percent": mape,
        "pearson": pearson_corr,
        "stdev_mape_perc": stdev
    }

SAVE_STEPS = max(int(STEPS_PER_EPOCH / EVALS_PER_EPOCH), 10)
LOG_STEPS = max(int(SAVE_STEPS / 15), 1)
head_params = ["classifier", "pooler", "score"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(hp in n for hp in head_params)],
        "lr": LR,
    },
    {
        "params": [p for n, p in model.named_parameters() if any(hp in n for hp in head_params)],
        "lr": LR_HEAD,
    },
]
optimizer = torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=0.01)
training_args = TrainingArguments(
    output_dir="./models/dense_unit_regar",
    learning_rate=LR if not LORA else LR_LORA,
    lr_scheduler_type="cosine",
    per_device_train_batch_size=A_BATCH_ACT,
    gradient_accumulation_steps=GR_ACC_ACT,
    per_device_eval_batch_size=A_BATCH_ACT,
    num_train_epochs=NUM_EPOCHS,
    #max_grad_norm=10.0,
    logging_nan_inf_filter=False,
    weight_decay=0.01,
    eval_strategy="steps",
    eval_steps=SAVE_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    load_best_model_at_end=True,
    save_total_limit=3,
    metric_for_best_model="rmse_g",
    greater_is_better=False,
    dataloader_pin_memory=False,
    logging_steps=LOG_STEPS,
    bf16=True,
    warmup_steps=WARMUP_STEPS

)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tok_ds_train,
    eval_dataset=tok_ds_eval,
    processing_class=tokenizer,
    optimizers=(optimizer, None),
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    compute_loss_func=None if not L1_LOSS else smooth_l1_loss_func
)

trainer.train()
