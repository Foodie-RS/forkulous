from collections import deque
import json
import os
import readchar

from datasets import Dataset, DatasetDict
from mpmath import sigmoid
from sentence_transformers import CrossEncoder

DS_STORE_PATH="./datasets/training/dataset_hard_usda.hf"
DS_QUERY_PROMPT="query: "
DS_DOC_PROMPT="document: "
LLM_DS_PATH="./datasets/llm-annotated/dataset_hard_usda_to_foodon.json"
LLM_DS_PATH2="./datasets/llm-annotated/dataset_usda_to_foodon.json"
LLM_QUERY_COL="foodon"
LLM_DOC_COL="usda"
MODEL_NAME="./models/ce-ettin-usda-4/checkpoint-1200"
HARD_STORE_PATH="./datasets/ce-annotated/mine_hard_negs_ettin_usda.json"
HARD_STORE_PATH_HUMAN="./datasets/human-annotated/hard_examples_ettin.json"
SIMILARITY_MAP = {
    "match": 1.0,
    "partial match": 0.4,
    "marginal match": 0.15,
    "no match": 0.0
}

EXCLUDE_LABELS = ["unsure"]

if not os.path.exists(HARD_STORE_PATH):
    with open(HARD_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump([], f)


with open(HARD_STORE_PATH_HUMAN, "w", encoding="utf-8") as f:
    json.dump([], f)

ds_all = DatasetDict.load_from_disk(DS_STORE_PATH)
ds_t = ds_all["train"]
ds_t = ds_t.add_column("train", [True] * len(ds_t))
ds_e = ds_all["eval"]
ds_e = ds_e.add_column("train", [False] * len(ds_e))
ds = Dataset.from_list(ds_t.to_list() + ds_e.to_list())
f = open(LLM_DS_PATH, "r")
ds_llm = json.load(f)
f = open(LLM_DS_PATH2, "r")
ds2_llm = json.load(f)
ds_llm.extend(ds2_llm)

def mine():
    model = CrossEncoder(MODEL_NAME, num_labels=1)
    cnt = 0
    hard_examples = []

    for row in ds:
        cnt+=1
        if cnt % 100 == 0:
            print(f"Processed {cnt}")
            f = open(HARD_STORE_PATH, "w")
            json.dump(hard_examples, f, indent=4)
        query_cleaned = row["query"][len(DS_QUERY_PROMPT):]
        doc_cleaned = row["document"][len(DS_DOC_PROMPT):]
        verdict = "none"
        reasoning = ""
        for llm_row in ds_llm:
            if llm_row[LLM_QUERY_COL] == query_cleaned and llm_row[LLM_DOC_COL] == doc_cleaned:
                verdict = llm_row["verdict"]
                reasoning = llm_row["reasoning"]
        if verdict == "none":
            print(f"Not found: \"{query_cleaned}\"->\"{doc_cleaned}\", skipping")
            continue
        elif verdict == "unsure":
            continue
        score_llm = SIMILARITY_MAP[verdict]
        score = sigmoid(model.predict((row["query"], row["document"])))
        score_diff = abs(score-score_llm)
        if score_diff > 0.35:
            hard_examples.append({
                "query": query_cleaned,
                "document": doc_cleaned,
                "ce_score": float(score),
                "llm_score": float(score_llm),
                "llm_reasoning": reasoning,
                "train": row["train"]
            })
    f = open(HARD_STORE_PATH, "w")
    json.dump(hard_examples, f, indent=4)

def human_eval():
    f = open(HARD_STORE_PATH, "r")
    items = json.load(f)
    processed = deque()
    unprocessed = deque(items)
    while True:
        with open(HARD_STORE_PATH_HUMAN, "r+", encoding="utf-8") as f:
            existing_data = json.load(f)
            results = []
            while len(processed) > 5:
                (it, ch) = processed.popleft()
                result = {
                    LLM_QUERY_COL: it["query"],
                    LLM_DOC_COL: it["document"],
                    "train": it["train"],
                }
                if ch == "d":
                    result["verdict"] = "delete",
                elif ch == "m":
                    result["verdict"] = "match",
                elif ch == "p":
                    result["verdict"] = "partial match"
                elif ch == "b":
                    result["verdict"] = "marginal match",
                elif ch == "n":
                    result["verdict"] = "no match"
                results.append(result)
            existing_data.extend(results)
            f.seek(0)
            json.dump(existing_data, f, indent=4, ensure_ascii=False)
            f.truncate()
        row = unprocessed.pop()
        print("============================")
        print(f"Query: {row["query"]}")
        print(f"Document: {row["document"]}")
        print(f"CE Score: {row["ce_score"]}")
        print(f"LLM Score: {row["llm_score"]}")
        print(f"LLM Reasoning: {row["llm_reasoning"]}")
        print()
        ch = readchar.readchar()
        while True:
            if ch in ["d", "m", "p", "b", "n"]:
                if ch == "d":
                    print("Deleting")
                elif ch == "m":
                    print("Match")
                elif ch == "p":
                    print("Partial Match")
                elif ch == "b":
                    print("Barely Matches")
                elif ch == "n":
                    print("No Match")
                processed.append((row, ch))
                break
            elif ch == "u":
                if len(processed) > 0:
                    (it, _) = processed.pop()
                    unprocessed.append(row)
                    unprocessed.append(it)
                    break
                else:
                    print("Queue is empty")

human_eval()
