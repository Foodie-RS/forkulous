from csv import DictReader
import json
import os
import random

from datasets import Dataset, DatasetDict
import ingredient_parser
from mpmath import sigmoid
from sentence_transformers import CrossEncoder, SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, base
from sentence_transformers.sentence_transformer.evaluation import BinaryClassificationEvaluator, EmbeddingSimilarityEvaluator
from sentence_transformers.sentence_transformer.losses import CoSENTLoss
from sentence_transformers.util import semantic_search

##### MODEL CONFIGURATION
BASE_MODEL_NAME="sentence-transformers/multi-qa-mpnet-base-cos-v1"
MODEL_OUTPUT="./models/be-freetext-usda-v2/checkpoint-1230"

BE_Q_PROMPT="ingredient: "
BE_D_PROMPT="usda: "

##### TRAINING ARGUMENTS
LR=5e-5
LR_STRAT="cosine"
BATCH_SIZE=32
EPOCHS=5
WARMUP_RATIO = 0.1
EVAL_COUNT = 10

TRAIN_DATA="datasets/training/usda_newprompt.train.json"
EVAL_DATA="datasets/training/usda_newprompt.eval.json"

###### DISTILLATION FROM CROSS ENCODER CONFIG
DO_CE_DISTILL=True
RETRIEVE_MODEL="./models/be-freetext-usda-fine_fulllines-veryhard-066-2/checkpoint-745"
PARSE_BEFORE_RETRIEVAL=False
DST_Q_PROMPT="ingredient: "
DST_D_PROMPT="usda: "

PARSE_DATASET_CSV_FOR_EXCLUSION=True

CE_NAME_FOR_DISTILLATION = "./models/ce-ettin-freetext2usda/checkpoint-980"
CE_USES_LOGIT=True
CE_Q_PROMPT="query: "
CE_D_PROMPT="document: "

DST_MINE_K=50000
DST_LOG_STEP=200
DST_SEARCH_TOP_K=10
DST_SCORE_DIFF_MIN=0.2

DST_QUERY_CSV="./datasets/database-dumps/recipe-lines-100k.csv"
DST_QUERY_COL="line"
DST_DOC_CSV="./datasets/database-dumps/all-foodbase-descriptions.csv"
DST_DOC_COL="description"


base_model = SentenceTransformer(BASE_MODEL_NAME)

def mine_data(existing_data:Dataset) -> Dataset:
    print("Collecting mining data...")
    excluded = 0
    # exclude all previously seen ingredient lines
    exclusion = []
    for it in existing_data["query"]:
        it = it[len(BE_Q_PROMPT):]
        if PARSE_DATASET_CSV_FOR_EXCLUSION:
            parsed = ingredient_parser.parse_ingredient(it)
            if len(parsed.name) > 0 and len(parsed.name[0].text.strip()) > 0:
                exclusion.append(parsed.name[0].text.strip().lower())
                continue
        exclusion.append(it.lower())
    with open(DST_QUERY_CSV, "r") as query_file:
        query_reader = DictReader(query_file)
        all_queries = [(ix, row[DST_QUERY_COL]) for ix,row in enumerate(query_reader)]
    query_count=len(all_queries)
    with open(DST_DOC_CSV, "r") as doc_file:
        doc_reader = DictReader(doc_file)
        all_docs = [row[DST_DOC_COL] for row in doc_reader]
    retrieve_mdl = SentenceTransformer(RETRIEVE_MODEL)
    rerank_mdl = CrossEncoder(CE_NAME_FOR_DISTILLATION)

    print("Running doc embed...")
    doc_embd = retrieve_mdl.encode([f"{DST_D_PROMPT}{d}" for d in all_docs], convert_to_numpy=False)
    print("Done embedding. Mining starting now.")

    mined_tuples = []
    prev_cnt = 0
    while len(mined_tuples) < DST_MINE_K:
        if len(all_queries)==0:
            print("Mine exhausted! Stopping.")
            break
        if len(mined_tuples) - prev_cnt > DST_LOG_STEP:
            mined = len(mined_tuples)
            total_examined = query_count - len(all_queries)
            print(json.dumps(mined_tuples[-5:-1], indent=4))
            print(f"Mined {mined}/{DST_MINE_K}")
            print(f"Pool: {len(all_queries)}/{query_count}")
            print(f"Excluded: {excluded}")
            print(f"Total examined: {total_examined}")
            print(f"Predicted number needed for {DST_MINE_K}: {(total_examined * DST_MINE_K)/mined}")
            prev_cnt = len(mined_tuples)

        (query_ix, query) = random.choice(all_queries)
        all_queries.remove((query_ix, query))
        q_txt = query
        exclusion_str = query
        if PARSE_BEFORE_RETRIEVAL or PARSE_DATASET_CSV_FOR_EXCLUSION:
            try:
                ingr_parsed = ingredient_parser.parse_ingredient(query)
            except Exception as e:
                print(e)
                continue
            if len(ingr_parsed.name) == 0 or ingr_parsed.name[0].text.strip() == "":
                continue
            if PARSE_DATASET_CSV_FOR_EXCLUSION:
                exclusion_str = ingr_parsed.name[0].text.strip()
            if PARSE_BEFORE_RETRIEVAL:
                q_txt = ingr_parsed.name[0].text.strip()
        if not query or not q_txt or not exclusion_str:
            continue

        if exclusion_str.lower() in exclusion:
            excluded += 1
            continue

        exclusion.append(exclusion_str.lower())

        query_vec = retrieve_mdl.encode(f"{DST_Q_PROMPT}{q_txt}")

        retr_res = semantic_search(query_vec, doc_embd, top_k=DST_SEARCH_TOP_K)[0]
        retr_res_mapped = [(all_docs[res["corpus_id"]], res["score"]) for res in retr_res]

        rrk_res = rerank_mdl.rank(f"{CE_Q_PROMPT}{query}", [f"{CE_D_PROMPT}{doc}" for doc,_ in retr_res_mapped])
        rrk_res_mapped = [(retr_res_mapped[res["corpus_id"]][0], retr_res_mapped[res["corpus_id"]][1], res["score"]) for res in rrk_res]

        for nam,tf_score,ce_score in rrk_res_mapped:
            if not nam:
                continue
            if CE_USES_LOGIT:
                ce_score=sigmoid(ce_score)
            if abs(tf_score-ce_score) > DST_SCORE_DIFF_MIN:
            #if True:
                exclusion.append(q_txt)
                mined_tuples.append({
                    "query":f"{BE_Q_PROMPT}{query}",
                    "document":f"{BE_D_PROMPT}{nam}",
                    "score":float(ce_score),
                    "tf_query":q_txt,
                    "tf_score": tf_score,
                    "exclusion": exclusion_str
                })
    print(f"Mining done! Mined {len(mined_tuples)}")
    return Dataset.from_list(mined_tuples).remove_columns(["tf_query", "tf_score", "exclusion"])

with open(TRAIN_DATA, "r") as f:
    train_data_json = json.load(f)

with open(EVAL_DATA, "r") as f:
    eval_data_json = json.load(f)

train_dataset = Dataset.from_list(train_data_json).map(lambda row: {
    "query": f"{BE_Q_PROMPT}{row["query"]}",
    "document": f"{BE_D_PROMPT}{row["document"]}",
    "score": row["score"]
}).shuffle(42)
eval_dataset = Dataset.from_list(eval_data_json).map(lambda row: {
    "query": f"{BE_Q_PROMPT}{row["query"]}",
    "document": f"{BE_D_PROMPT}{row["document"]}",
    "score": row["score"]
}).shuffle(42)

loss = CoSENTLoss(model=base_model)
evaluator = EmbeddingSimilarityEvaluator(
    scores=eval_dataset["score"],
    sentences1=eval_dataset["query"],
    sentences2=eval_dataset["document"],
    main_similarity="cosine"
)
num_steps = int((len(train_dataset) / BATCH_SIZE) * EPOCHS)
eval_steps = int(num_steps / EVAL_COUNT)
args = SentenceTransformerTrainingArguments(
    output_dir=MODEL_OUTPUT,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    lr_scheduler_type=LR_STRAT,
    eval_strategy="steps",
    eval_steps=eval_steps,
    save_strategy="steps",
    save_steps=eval_steps,
    weight_decay=0.01,
    warmup_steps=WARMUP_RATIO*EPOCHS*len(train_dataset)/BATCH_SIZE,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="spearman_cosine",
    greater_is_better=True,
    save_total_limit=3,
)

trainer = SentenceTransformerTrainer(
    model=base_model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
    evaluator=evaluator
)

print("Starting Training...")
trainer.train()
