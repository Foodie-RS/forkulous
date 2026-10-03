import csv
import json
import numpy as np

import pint
from sentence_transformers import SentenceTransformer
from usearch.index import Index
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit

GOOD_VOLUME_UNITS = {"cup", "cups", "cubic inch", "cubic inches", "quart", "quarts", "pint", "pints", "ml", "milliliter", "milliliters", "l", "liter", "liters"}
with open("./fdc_portions.json", "r") as f:
    unit_corpus = {int(x):y for x,y in json.load(f).items()}

with open("./all-foodbase-descriptions.csv", "r") as f:
    rows = csv.DictReader(f)
    fdc_dict = {int(row["fdc_id"]):row["description"] for row in rows}
    fdc_ids = np.array([int(x) for x in fdc_dict.keys()])
    fdc_descs = [fdc_dict[x] for x in fdc_ids]

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)

embeddings = embedder.encode(fdc_descs, convert_to_numpy=True, show_progress_bar=True)
N, D = embeddings.shape
keys = np.arange(N, dtype=np.uint64)
index = Index(ndim=D, metric="cos", dtype="f16")
index.add(keys, embeddings)

clustering = index.cluster(
    min_count=10,  # Minimale Clustergröße
    max_count=100  # Maximale Clustergröße
)

centroid_keys, _ = clustering.centroids_popularity
cluster_ids = np.empty(len(embeddings), dtype=np.uint64)

for c_key in centroid_keys:
    member_indices = clustering.members_of(c_key)
    cluster_ids[member_indices] = c_key

# Sicherheitscheck: Muss exakt die Anzahl deiner Datenpunkte (N) sein!
assert len(cluster_ids) == len(embeddings)

gss = GroupShuffleSplit(n_splits=1, test_size=0.13, random_state=42)
train_idx, eval_idx = next(gss.split(embeddings, groups=cluster_ids))
train_ids = fdc_ids[train_idx]
eval_ids = fdc_ids[eval_idx]

train_data = {
    "label": [],
    "text": []
}

eval_data = {
    "label": [],
    "text": []
}

def generate_nonstandard():
    for fdc_id,units in unit_corpus.items():
        desc = fdc_dict[fdc_id]
        train = fdc_id in train_ids
        for unit in units:
            if unit["unit_name"] in ["","cup", "cups", "oz", "ounce", "ounces", "fl oz", "fluid ounce", "lb", "lbs", "pounds", "pound"]:
                continue
            text=f"""FOOD: {desc}
UNIT: {unit["unit_name"]}
COMMENTS: {"none" if len(unit["comments"]) == 0 else f"[{", ".join(unit["comments"])}]"}"""
            label = unit["gram_weight"]
            if train:
                train_data["label"].append(label)
                train_data["text"].append(text)
            else:
                eval_data["label"].append(label)
                eval_data["text"].append(text)
    with open("./fdc_unit_reg.train.json", "w") as f:
        json.dump(train_data, f)

    with open("./fdc_unit_reg.eval.json", "w") as f:
        json.dump(eval_data, f)

def generate_unspecified():
    for fdc_id,units in unit_corpus.items():
        desc = fdc_dict[fdc_id]
        train = fdc_id in train_ids
        for unit in units:
            if unit["unit_name"] not in ["", "serving", "servings"]:
                continue
            text=f"""FOOD: {desc}
COMMENTS: {"none" if len(unit["comments"]) == 0 else f"[{", ".join(unit["comments"])}]"}"""
            label = unit["gram_weight"]
            if train:
                train_data["label"].append(label)
                train_data["text"].append(text)
            else:
                eval_data["label"].append(label)
                eval_data["text"].append(text)
    with open("./fdc_unit_reg_unspec.train.json", "w") as f:
        json.dump(train_data, f)

    with open("./fdc_unit_reg_unspec.eval.json", "w") as f:
        json.dump(eval_data, f)

def generate_density():
    ureg = pint.UnitRegistry()
    for fdc_id,units in unit_corpus.items():
        desc = fdc_dict[fdc_id]
        train = fdc_id in train_ids
        for unit in units:
            if unit["unit_name"] not in GOOD_VOLUME_UNITS:
                continue
            try:
                un = 1 * ureg(unit["unit_name"])
                un_ml = float(un.to("ml").m)
            except:
                continue
            text=f"""FOOD: {desc}
COMMENTS: {"none" if len(unit["comments"]) == 0 else f"[{", ".join(unit["comments"])}]"}"""
            label = (unit["gram_weight"]/un_ml)
            if label > 10:
                continue
            if text in train_data["text"] or text in eval_data["text"]:
                break
            if train:
                train_data["label"].append(label)
                train_data["text"].append(text)
            else:
                eval_data["label"].append(label)
                eval_data["text"].append(text)
            break
    with open("./fdc_unit_reg_dense.train.json", "w") as f:
        json.dump(train_data, f)

    with open("./fdc_unit_reg_dense.eval.json", "w") as f:
        json.dump(eval_data, f)

generate_density()

print(f"Training: {len(train_data["label"])}")
print(f"Eval: {len(eval_data["label"])}")
print(f"Ratio: {len(eval_data["label"])/(len(eval_data["label"]) + len(train_data['label']))}")
