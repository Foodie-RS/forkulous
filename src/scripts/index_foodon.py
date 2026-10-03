from csv import DictReader
import csv
from datetime import datetime
from itertools import islice
import json
import os
import pickle
from typing import Dict, List, NamedTuple, Set, Tuple
import typing

from mpmath import sigmoid
from rdflib import Graph, Literal, URIRef
from sentence_transformers import CrossEncoder, SentenceTransformer
import sentence_transformers
from sentence_transformers.util import semantic_search
from torch import Tensor, torch

#model_usda = SentenceTransformer("models/be-usda-4")
#model_freetext = SentenceTransformer("models/be-freetext-4")

#ce_model_q_foodon = CrossEncoder("models/ce-minilm-q_foodon-4")

usda_descriptions_file = "datasets/all-foodbase-descriptions.csv"
usda_embeddings_file = "embeddings/usda_embeddings.dat"
foodon_embeddings_file_usda = "embeddings/foodon_usda_embeddings.dat"
foodon_embeddings_file_freetext= "embeddings/foodon_freetext_embeddings.dat"

foodon_embeddings_usda:List[Tuple[URIRef,str,Tensor]] = []
foodon_embeddings_freetext:List[Tuple[URIRef, str,Tensor]] = []
usda_embeddings:Dict[str,Tensor] = {}

foodon_to_usda_file = "embeddings/foodon_usda_relations.json"
foodon_usda_relations:Dict[str, List[Tuple[str, float]]] = {}

usda_names:Dict[int,str] = {}

g = Graph()
g.parse("foodon.owl", format="xml")

rootNode = URIRef("http://purl.obolibrary.org/obo/FOODON_00002403")

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

class NamedURI(NamedTuple):
    ref: URIRef
    name: str

def subclass_relations() -> Dict[URIRef, List[URIRef]]:
    query = """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX FOODON: <http://purl.obolibrary.org/obo/FOODON_>

    SELECT DISTINCT ?cls ?subClass
    WHERE {
        ?subClass a owl:Class .
        ?cls rdfs:label ?className .
        ?subClass rdfs:subClassOf ?cls .
        ?subClass rdfs:label ?subClassName .
        FILTER (!STRSTARTS(?subClassName, "obsolete"))
        FILTER (?subClass != ?cls)
        }
        """
    dct: Dict[URIRef, List[URIRef]] = {}
    for row in g.query(query):
        if row[0] in dct:
            dct[row[0]].append(row[1])
        else:
            dct[row[0]] = [row[1]]
    return dct


def names() -> Dict[URIRef, str]:
    query = """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT DISTINCT ?uri ?className
    WHERE {
        ?uri rdfs:label ?className .
    }
    """
    dct = {}
    for row in g.query(query):
        dct[row[0]] = row[1].value
    return dct


def descs() -> Dict[URIRef, str]:
    query = """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX obo: <http://purl.obolibrary.org/obo/IAO_>
    SELECT DISTINCT ?uri ?desc
    WHERE {
        ?uri obo:0000115 ?desc .
    }
    """
    dct = {}
    for row in g.query(query):
        if row[1].__class__ != Literal:
            print(row[1])
        else:
            dct[row[0]] = row[1].value
    return dct

def load_usda_names():
    global usda_names
    with open(usda_descriptions_file, 'r') as f:
        reader = DictReader(f)
        for row in reader:
            usda_names[row["fdc_id"]] = row["description"]

_names = names()
_subclass_relations: Dict[URIRef, List[URIRef]] = subclass_relations()
_descs = descs()
print(
    f"Got {len(_names)} names, {len(_descs)} Descriptions and {len(_subclass_relations)} relations"
)

def load_embeddings_new():
    global foodon_embeddings_freetext, foodon_embeddings_usda, usda_embeddings
    embeddings_file = open(foodon_embeddings_file_usda, "rb")
    foodon_embeddings_usda = pickle.load(embeddings_file)
    embeddings_file = open(foodon_embeddings_file_freetext, "rb")
    foodon_embeddings_freetext = pickle.load(embeddings_file)
    embeddings_file = open(usda_embeddings_file, "rb")
    usda_embeddings = pickle.load(embeddings_file)

def generate_embeddings_usda():
    counter = 0
    for chnk in chunk(usda_names.items(), 16):
        if counter % 10 == 0 and counter > 0:
            print(f"USDA: Processed {counter * 16} items")
        counter += 1
        chnk = list(chnk)
        nam = [f"usda: {desc}" for _,desc in chnk]
        embd = model_usda.encode(nam)
        for i, (fdc_id, _) in enumerate(chnk):
            usda_embeddings[fdc_id] = embd[i]
    embeddings_file = open(usda_embeddings_file, "wb")
    pickle.dump(usda_embeddings, embeddings_file)

def generate_embeddings_foodon():
    all_names = recursive_names(NamedURI(rootNode, ""), [])
    print(f"FoodOn: Total names: {len(all_names)}")
    print(next(iter(all_names)))
    counter = 0
    for chnk in chunk(all_names, 16):
        if counter % 10 == 0 and counter > 0:
            print(f"FoodOn: Processed {counter * 16} items")
        counter += 1
        chnk = list(chnk)
        nam_u = [f"foodon: {name}" for _, name in chnk]
        nam_f = [f"document: {name}" for _, name in chnk]
        embd_u = model_usda.encode(nam_u)
        embd_f = model_freetext.encode(nam_f)
        for i, (ref,nam) in enumerate(chnk):
            foodon_embeddings_usda.append((ref, nam, embd_u[i]))
            foodon_embeddings_freetext.append((ref, nam, embd_f[i]))

    embedding_file = open(foodon_embeddings_file_usda, "wb")
    pickle.dump(foodon_embeddings_usda, embedding_file)
    embedding_file = open(foodon_embeddings_file_freetext, "wb")
    pickle.dump(foodon_embeddings_freetext, embedding_file)


def recursive_names(root:NamedURI, path:List[NamedURI]) -> Set[Tuple[URIRef,str]]:
    subclasses = [] if root.ref not in _subclass_relations else _subclass_relations[root.ref]
    names = set()
    path_name = f" (path: {": ".join([it.name for it in path])})"
    for cls in subclasses:
        skip = False
        for it in path:
            if it.ref == cls:
                skip = True
                break
        if skip:
            continue
        if cls in _names:
            nam = _names[cls]
            names.add((cls, f"{nam} {path_name}"))
        else:
            continue
        cls_ref = NamedURI(cls, nam)
        if cls in _descs:
            names.add((cls, f"{_descs[cls]}"))
        names = names.union(recursive_names(cls_ref, path + [cls_ref]))
    return names

def generate_foodon_relations():
    if not os.path.exists(foodon_to_usda_file):
        with open(foodon_to_usda_file, "w", encoding="utf-8") as f:
            json.dump([], f)
    foodon_tns = torch.stack([torch.from_numpy(v) for _,_,v in foodon_embeddings_usda]);
    tuples_usda:List[Tuple[str, Tensor]] = list(usda_embeddings.items())
    embeddings_usda = torch.stack([torch.from_numpy(v) for _,v in tuples_usda])
    best_usda_items:List[Dict] = []
    print("Starting semantic search")
    start_time = datetime.now()
    tf_hits = semantic_search(foodon_tns, embeddings_usda,top_k=100)
    end_time = datetime.now()
    print(f"Took {(end_time-start_time).total_seconds()} seconds.")
    for i, (ref, nam, _) in enumerate(foodon_embeddings_usda):
        if i % 100 == 0:
            print(f"Processed {i}/{len(foodon_embeddings_usda)} items")
            with open(foodon_to_usda_file, "r+", encoding="utf-8") as f:
                existing_data = json.load(f)
                existing_data.extend(best_usda_items)
                f.seek(0)
                json.dump(existing_data, f, indent=4, ensure_ascii=False)
                f.truncate()
                best_usda_items.clear()

        tf_hits_mapped = [{
            "fdc_id": int(tuples_usda[row["corpus_id"]][0]),
            "tf_score": float(row["score"])
        } for row in tf_hits[i]]
        tf_hits_txt = [f"document: {usda_names[str(row["fdc_id"])]}" for row in tf_hits_mapped]
        ce_hits = ce_model_q_foodon.rank(f"query: {nam}", tf_hits_txt, top_k=5)
        ce_hits_mapped = [{
            "fdc_id": int(tf_hits_mapped[row["corpus_id"]]["fdc_id"]),
            "description": usda_names[str(tf_hits_mapped[row["corpus_id"]]["fdc_id"])],
            "tf_score": float(tf_hits_mapped[row["corpus_id"]]["tf_score"]),
            "score": float(row["score"]),
            "score_sig": float(sigmoid(row["score"]))
        } for row in ce_hits]
        best_usda_items.append({
            "ref": str(ref),
            "name": nam,
            "results": ce_hits_mapped
        })

    with open(foodon_to_usda_file, "r+", encoding="utf-8") as f:
        existing_data = json.load(f)
        existing_data.extend(best_usda_items)
        f.seek(0)
        json.dump(existing_data, f, indent=4, ensure_ascii=False)
        f.truncate()


#load_usda_names()
#generate_embeddings_foodon()
#generate_embeddings_usda()
#load_embeddings_new()
#generate_foodon_relations()
names_rec = recursive_names(NamedURI(rootNode, ""), [])
count = 0
with open("./datasets/database-dumps/all-foodon-descriptions.csv", "x") as output_file:
    writer = csv.DictWriter(output_file, ["id","uri", "description"])
    for ref, name in names_rec:
        writer.writerow({
            "id": count,
            "uri": ref,
            "description": name
        })
        count+=1
