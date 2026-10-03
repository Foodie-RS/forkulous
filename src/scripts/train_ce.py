import json

from datasets import Dataset, DatasetDict
from peft import LoraConfig, TaskType
from sentence_transformers import CrossEncoder, CrossEncoderTrainer, CrossEncoderTrainingArguments
from sentence_transformers.cross_encoder.evaluation import CrossEncoderCorrelationEvaluator

TRAIN_DATA="datasets/training/usda_newprompt.train.json"
EVAL_DATA="datasets/training/usda_newprompt.eval_disjoint.json"

MODEL_NAME="cross-encoder/ettin-reranker-17m-v1"
#MODEL_NAME="Qwen/Qwen3-Reranker-4B"
MODEL_OUTPUT="./models/ce-ettin-freetext2usda_tiny_b"
ACTION_PROMPT=""

QUERY_PROMPT="query: "
DOCUMENT_PROMPT="document: "

LR=5e-5
USE_LORA=False
LR_STRAT="linear"
EFFECTIVE_BATCH_SIZE=2
ACTUAL_BATCH_SIZE=2
EPOCHS=5
EVAL_PERIOD=200

gr_acc_steps = EFFECTIVE_BATCH_SIZE / ACTUAL_BATCH_SIZE
if not gr_acc_steps.is_integer():
   raise ValueError("Effective batch size must be multiple of actual size")

gr_acc_steps = int(gr_acc_steps)
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
    lora_dropout=0.05
)

with open(TRAIN_DATA, "r") as f:
    train_data_json = json.load(f)

with open(EVAL_DATA, "r") as f:
    eval_data_json = json.load(f)

train_dataset = Dataset.from_list(train_data_json).map(lambda row: {
    "query": f"{QUERY_PROMPT}{row["query"]}",
    "document": f"{DOCUMENT_PROMPT}{row["document"]}",
    "score": row["score"]
}).shuffle(42)
eval_dataset = Dataset.from_list(eval_data_json).map(lambda row: {
    "query": f"{QUERY_PROMPT}{row["query"]}",
    "document": f"{DOCUMENT_PROMPT}{row["document"]}",
    "score": row["score"]
}).shuffle(42)


val_pairs = list(zip(eval_dataset["query"], eval_dataset["document"]))
val_scores = eval_dataset["score"]

evaluator = CrossEncoderCorrelationEvaluator(
    #prompt_name="action",
    sentence_pairs=val_pairs,
    scores=val_scores,
)

model = CrossEncoder(MODEL_NAME, num_labels=1, max_length=128,
prompts={
    "action": ACTION_PROMPT
}
)
print(evaluator.__call__(model))
if USE_LORA:
    model.add_adapter(lora_config)
training_args = CrossEncoderTrainingArguments(
    output_dir=MODEL_OUTPUT,
    num_train_epochs=EPOCHS,
    prompts=ACTION_PROMPT,
    per_device_train_batch_size=EFFECTIVE_BATCH_SIZE if not USE_LORA else ACTUAL_BATCH_SIZE,
    per_device_eval_batch_size=EFFECTIVE_BATCH_SIZE if not USE_LORA else ACTUAL_BATCH_SIZE,
    gradient_accumulation_steps=gr_acc_steps if USE_LORA else 1,
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=EVAL_PERIOD,
    save_steps=EVAL_PERIOD,
    warmup_ratio=0.1,
    lr_scheduler_type=LR_STRAT,
    learning_rate=LR,
    weight_decay=0.1,
    dataloader_pin_memory=False,
    logging_steps=10,
    bf16=True,
    save_total_limit=3,
    metric_for_best_model="spearman",
    load_best_model_at_end=True,
)

trainer = CrossEncoderTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    evaluator=evaluator
)

trainer.train()
