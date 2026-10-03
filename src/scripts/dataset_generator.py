from csv import DictReader
import json
import os
import random

from datasets import Dataset
import ingredient_parser
from sentence_transformers import SentenceTransformer
from usearch.index import Index
from dotenv import load_dotenv
from api import ingredient_search


def llm_preprocess(source_file, dst_file, existing_tuples_file, tuple_count):
    ing_api = ingredient_search.load2(
        be_model="./models/be-freetext-usda-fine_fulllines-big-bettersplit/checkpoint-3000",
        ce_model="./models/ce-ettin-freetext2usda_base-bettersplit/checkpoint-252",
        unit_ranker="sentence-transformers/all-MiniLM-L6-v2",
        index_file="./embeddings/index_u2f.dat",
        corpus_file_csv="./datasets/database-dumps/all-foodbase-descriptions.csv",
        id_column="fdc_id",
        corpus_doc_column="description",
        be_query_prompt="ingredient: ",
        be_doc_prompt="usda: ",
        action_prompt=""
    )

    with open(existing_tuples_file, "r") as f:
        contents = json.load(f)
        existing = set([row["ingredient"] for row in contents])

    data = []
    try:
        with open(source_file, "r") as f:
            drd = DictReader(f)
            for row in drd:
                if row["line"] in existing:
                    continue
                items = ing_api.search_with_hard(row["line"])
                data.extend(items)
                if len(data) >= tuple_count:
                    break
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(e)
    with open(dst_file, "w") as f:
        json.dump(data, f)

def llm_postprocess(src_file, train_file, eval_file, eval_ratio):
    verdict_map = {
        "match": 1.0,
        "near match": 0.9,
        "mostly match": 0.75,
        "partial match": 0.4,
        "marginal match": 0.1,
        "no match": 0.0,
        "unsure": -1,
        "invalid": -1
    }
    print("Loading and processing data...")
    with open(src_file, "r") as f:
        data = json.load(f)
    mapped_data = {}
    nonparsed = 0
    discarded = 0
    for count,row in enumerate(data):
        if count % 500 == 0:
            print(f"Processed {count} items (discarded {discarded}, non-parsed {nonparsed})")
        score = verdict_map[row["verdict"]]
        if score < 0:
            discarded += 1
            continue
        item = {
            "query": row["ingredient"],
            "document": row["usda"],
            "score": score
        }
        parsed = ingredient_parser.parse_ingredient(row["ingredient"])
        if len(parsed.name) > 0 and len(parsed.name[0].text.strip()) > 0:
            mapped_data.setdefault(parsed.name[0].text.strip().lower(), []).append(item)
        else:
            nonparsed += 1
            mapped_data.setdefault(row["ingredient"].strip().lower(), []).append(item)
    keys = list(mapped_data.keys())
    print(f"Total of {len(keys)} unique ingredients")
    eval_keys = random.sample(keys, k=int(eval_ratio*len(keys)))
    train_set = [tup for key, it in mapped_data.items() if key not in eval_keys for tup in it]
    eval_set = [tup for key, it in mapped_data.items() if key in eval_keys for tup in it]
    print(f"Created train set: {len(train_set)} tuples, eval set: {len(eval_set)} tuples")
    with open(train_file, "w") as f:
        json.dump(train_set, f)
    with open(eval_file, "w") as f:
        json.dump(eval_set, f)

def analyse_data(train, eval, out_set):
    with open(train, "r") as f:
        train_data = json.load(f)
    with open(eval, "r") as f:
        eval_data = json.load(f)
    train_docs = set([row["document"] for row in train_data])
    eval_docs = set([row["document"] for row in eval_data])
    intersect = eval_docs.intersection(train_docs)
    print(f"Unique train docs: {len(train_docs)}, eval docs: {len(eval_docs)}")
    print(f"Intersection: {len(intersect)} ({len(eval_docs)-len(intersect)} only in eval set)")
    eval_tuples_new = [row for row in eval_data if row["document"] not in intersect]
    with open(out_set, "w") as f:
        json.dump(eval_tuples_new, f)

def get_llm_queries(src_file) -> set[str]:
    with open(src_file, "r") as f:
        data = json.load(f)
    queries = set()
    for row in data:
        queries.add(row["ingredient"].lower())
    return queries

def api():
    load_dotenv()
    ing_api = ingredient_search.load2(
        be_model="./models/be-freetext-usda-v2/checkpoint-1230",
        ce_model="./models/ce-ettin-freetext2usda_large_v2/checkpoint-1500",
        index_file="./embeddings/foodbase-embeddings.dat",
        corpus_file_csv="./datasets/database-dumps/all-foodbase-descriptions.csv",
        id_column="fdc_id",
        corpus_doc_column="description",
        unit_ranker="sentence-transformers/all-MiniLM-L6-v2",
        conninfo=os.environ["DATABASE_URL"]
    )
    return ing_api

def do_data_mine(lines_file,llm_processed_file, out_file):
    ing_api = api()
    llm_queries = get_llm_queries(llm_processed_file)
    query_list = []
    with open(lines_file, "r") as f:
        drd = DictReader(f)
        for row in drd:
            if row["line"].lower() in llm_queries:
                continue
            query_list.append(row["line"])
    triples = ing_api.gen_hard_triples(query_list=query_list, tgt_file=out_file)
    print("Write complete")

def data_mine_postprocess(mined_file, train_out_file, eval_out_file):
    eval_ratio = 0.2
    epsilon = 0.08
    min_dist = 0.3
    with open(mined_file, "r") as f:
        data:list = json.load(f)
    model = SentenceTransformer("./models/be-freetext-usda-v2/checkpoint-1230")
    random.seed(42)
    random.shuffle(data)
    train_ix = Index(ndim=model.get_embedding_dimension(), metric="cos")
    eval_ix = Index(ndim=model.get_embedding_dimension(), metric="cos")
    train_triples = []
    eval_triples = []
    eval_docs = set()
    t_ix_init = False
    e_ix_init = False
    seen = 0
    deciding_factors = {
        "doc_already_eval": 0,
        "eval_ix_close": 0,
        "train_ix_close": 0,
        "random": 0,
        "e_random_retained": 0,
        "t_random_retained": 0,
        "eval_full": 0,
        "eval_full_ret": 0,
        "e_despite_full": 0,
        "e_docs_but_full": 0
    }
    for (query, pos, neg) in data:
        seen += 1
        if seen % 100 == 0:
            print(f"{seen}/{len(data)}")
            print(f"{len(eval_triples)} eval triples")
            print(f"{(len(eval_triples) / seen):.4f} momentary eval ratio")
            print(f"{(len(eval_triples) / len(data)):.4f} total eval ratio")
            print(deciding_factors)
        epsilon_adj = epsilon * (1-(seen/len(data)))
        eval_full = (len(eval_triples) / seen) > (eval_ratio - epsilon_adj)
        if pos["text"].lower() in eval_docs:
            eval_triples.append({
                    "query": query,
                    "pos": pos["text"],
                    "neg": neg["text"],
                    "margin": pos["score"] - neg["score"]
            })
            if eval_full:
                deciding_factors["e_docs_but_full"] += 1
            else:
                deciding_factors["doc_already_eval"] += 1

        else:
            try_eval = (not eval_full) and random.random() >= 0.5
            embd = model.encode_query(query, convert_to_numpy=True)
            if t_ix_init and e_ix_init:
                train_dst = train_ix.search(embd, count=1, exact=True).distances.min()
                eval_dst = eval_ix.search(embd, count=1, exact=True).distances.min()
                if try_eval and train_dst < min_dist:
                    eval = eval_dst <= train_dst
                    if eval:
                        deciding_factors["e_random_retained"] += 1
                    else:
                        deciding_factors["train_ix_close"] += 1
                elif not try_eval and eval_dst < min_dist:
                    eval = eval_dst < train_dst
                    if eval:
                        if eval_full:
                            deciding_factors["e_despite_full"] += 1
                        else:
                            deciding_factors["eval_ix_close"] += 1
                    else:
                        if eval_full:
                            deciding_factors["eval_full_ret"] += 1
                        else:
                            deciding_factors["t_random_retained"] += 1
                else:
                    eval = try_eval
                    if eval_full:
                        deciding_factors["eval_full"] += 1
                    else:
                        deciding_factors["random"] += 1
            else:
                eval = try_eval
                if eval_full:
                    deciding_factors["eval_full"] += 1
                else:
                    deciding_factors["random"] += 1
            if eval:
                eval_ix.add(random.randint(0,9999999999999), embd)
                e_ix_init = True
                eval_triples.append({
                    "query": query,
                    "pos": pos["text"],
                    "neg": neg["text"],
                    "margin": pos["score"] - neg["score"]
                })
                eval_docs.add(pos["text"].lower())
            else:
                train_ix.add(random.randint(0,9999999999999), embd)
                t_ix_init = True
                train_triples.append({
                    "anchor": query,
                    "pos": pos["text"],
                    "neg": neg["text"],
                    "margin": pos["score"] - neg["score"]
                })
    print(deciding_factors)
    print(f"{len(eval_docs)} unique positive eval docs")
    train_ds = Dataset.from_list(train_triples)
    eval_ds = Dataset.from_list(eval_triples)
    train_ds.to_json(train_out_file)
    eval_ds.to_json(eval_out_file)

def mrr_eval_dataset(eval_ds_file, mrr_eval_out_set):
    _api = api()
    ds = Dataset.from_json(eval_ds_file)
    descs = {}
    c_reslv = {}
    corpus = {}
    for i,d in _api.corpus.items():
        if d in descs:
            c_reslv[i] = descs[d]
            continue
        corpus[f"d{i}"] = d
        descs[d] = i
        c_reslv[i] = i
    new_ds = {
        "queries": {},
        "relevant_docs": {},
        "corpus": corpus
    }
    for ix, query in enumerate(ds["query"]):
        if ix % 10 == 0:
            print(f"{ix}/{len(ds)}")
            with open(mrr_eval_out_set, "w") as f:
                json.dump(new_ds, f, indent=4)
        topk = [f"d{c_reslv[it["id"]]}" for it in _api.search_ingredient(query, {
            "be_top_k": 200,
            "ce_top_k": 200,
        })["search_result"] if it["confidence"] > 0.96]
        new_ds["queries"][f"q{ix}"] = query
        new_ds["relevant_docs"][f"q{ix}"] = topk
    with open(mrr_eval_out_set, "w") as f:
        json.dump(new_ds, f, indent=4)





#llm_postprocess("datasets/llm-annotated/freetext2usda_newprompt-2.json", "datasets/training/usda_newprompt.train.json", "datasets/training/usda_newprompt.eval.json", 0.2)
#analyse_data("datasets/training/usda_newprompt.train.json","datasets/training/usda_newprompt.eval.json","datasets/training/usda_newprompt.eval_disjoint.json"  )
#do_data_mine("./datasets/database-dumps/recipe-lines-100k.csv", "./datasets/llm-annotated/freetext2usda_newprompt-2.json", "./datasets/ce-annotated/triples_for_distillation_v2prompt.json")
#data_mine_postprocess("./datasets/ce-annotated/triples_for_distillation_v2prompt.json", "./datasets/training/v2_distilled_triples.train.jsonl", "./datasets/training/v2_distilled_triples.eval.jsonl")
#mrr_eval_dataset("./datasets/training/v2_distilled_triples.eval.jsonl", "./datasets/training/v2_distilled_mrr.eval.json")
#_api = api()
#while True:
#    inp = input("ingredient: ")
#    if inp == "":
#        inp = "1 cup heavy cream"
#    print(f"Search {inp}")
#    print(json.dumps(_api.search_ingredient(inp, {"parse":True}),default=lambda o:o.__dict__, indent=4))
