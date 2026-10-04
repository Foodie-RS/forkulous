from __future__ import annotations

import math

import json
from functools import reduce
from multiprocessing.dummy import Pool as ThreadPool
from typing import Any, NamedTuple

import pandas
from pandas.core.generic import DtypeArg
from pydantic import BaseModel
import regex


class Nutris(BaseModel):
    energy:float|None=None
    carbs:float|None=None
    fat:float|None=None
    protein:float|None=None
    salt:float|None=None
    sat_fat:float|None=None
    sugar:float|None=None
    fiber:float|None=None
    calculated:list[str]|None=None

    def toJSON(self):
        return json.dumps(self.to_dict())

    def to_dict(self) -> dict[str, float|None|list[str]]:
        nutris:dict[str, Any]={k:v for k,v in [("energy", self.energy), ("carbs", self.carbs), ("fat", self.fat), ("protein", self.protein), ("salt", self.salt), ("sugar", self.sugar), ("fiber", self.fiber), ("satfat", self.sat_fat)]}
        if self.calculated is not None and len(self.calculated) > 0:
            nutris["calculated"] = self.calculated
        else:
            nutris["calculated"] = None
        return nutris

    def multiply(self, factor:float) -> Nutris:
        return Nutris (
            energy=None if self.energy is None else self.energy * factor,
            carbs=None if self.carbs is None else self.carbs * factor,
            fat=None if self.fat is None else self.fat * factor,
            protein=None if self.protein is None else self.protein * factor,
            salt=None if self.salt is None else self.salt * factor,
            sugar=None if self.sugar is None else self.sugar * factor,
            fiber=None if self.fiber is None else self.fiber * factor,
            sat_fat=None if self.sat_fat is None else self.sat_fat * factor,
            calculated=self.calculated
        )
KCAL_TO_KJ_CNST = 4.184


FDC_CE_Q_PROMPT="query: "
FDC_CE_D_PROMPT="document: "
FDC_CE_ACTION_PROMPT=""

FDC_ENERGY_IDS=[1008,1062,2047,2048]

FDC_PROTEIN_IDS=[1003,1053]
FDC_FAT_IDS=[1004,1085]
FDC_CARB_IDS=[1005,1050,2039]

FDC_SALT_IDS=[1149]
FDC_SODIUM_IDS=[1093]

FDC_SATFAT_IDS=[1258,1326]
FDC_UNSATFAT_IDS=[[1292,1293]]

FDC_SUGAR_IDS=[1063,2000]
FDC_FIBER_IDS=[1079,2033]

RELEVANT_NUTRIS = FDC_ENERGY_IDS + FDC_PROTEIN_IDS + FDC_FAT_IDS + FDC_CARB_IDS + FDC_SALT_IDS + FDC_SODIUM_IDS + FDC_SATFAT_IDS + FDC_SUGAR_IDS + FDC_FIBER_IDS + [a for i in FDC_UNSATFAT_IDS for a in i]

def _fill_nutris(nutris:Nutris):
    calc = []
    if nutris.carbs is None:
        nutris.carbs = nutris.sugar
        calc.append("carbs")
    if nutris.fat is None:
        nutris.fat = nutris.sat_fat
        calc.append("fat")
    if nutris.energy is None and (nutris.fat is not None or nutris.protein is not None or nutris.carbs is not None):
        nutris.energy = ((nutris.fat or 0) * 9 + (nutris.protein or 0) * 4 + (nutris.carbs or 0) * 4) * KCAL_TO_KJ_CNST
        calc.append("energy")
    if len(calc) > 0:
        nutris.calculated = calc

UNIT_STOPWORDS=["NFS","bite","fun","thick","thin","of","and","or","with","without","medium","large","small", "on", "per", "drained", "calorie"]
UNIT_SEC_STOPWORDS=["oz","lb","lbs","cup", "cups"]
ORDINARY_WORDS=["medium","normal","regular"]
INFO_1_REGEX=regex.Regex(r"^(?P<a>[0-9]+)?(?P<c1>.*?)(?P<u>(?:\b[a-z][a-z0-9\-\']*\s*){1,2})(?P<c2>.*?)$")
INFO_2_REGEX=regex.Regex(r"^(?P<c1>.*?)(?P<u>(?:\b[a-z][a-z0-9\-\']*\s*){1,2})(?P<c2>.*?)$")
WORD_BOUNDARY_RX = regex.Regex(r"\b")
COMMENT_CLEAN_RX = regex.Regex(r"^\W*(.*?)\)?$")

class _NameCandidate(NamedTuple):
    name:str|None
    comments:list[str]
    modifier:int|None
    amount:int|None
    parser_confidence: int

def _resolve_unit_name(unit_name:str,unit_info_1:str,unit_info_2:str) -> _NameCandidate:
    if unit_info_2.isnumeric():
        modifier = int(unit_info_2)
        unit_info_2 = ""
    else:
        modifier = None
    if unit_name!="undetermined" and len(unit_name)>0:
        return _NameCandidate(unit_name, [x.strip() for x in [unit_info_1, unit_info_2] if len(x.strip()) > 0], modifier, None, 1)
    if unit_info_1 == "Quantity not specified":
        return _NameCandidate("", [unit_info_2] if len(unit_info_2.strip()) > 0 else [], modifier, None, 1)
    if len(unit_info_1) == 0:
        matches = INFO_2_REGEX.match(unit_info_2)
    else:
        matches = INFO_1_REGEX.match(unit_info_1)
    if matches is not None:
        amnt = None if len(unit_info_1) == 0 else matches.captures("a")
        unit = matches.captures("u")
        if len(unit) == 0:
            return _NameCandidate(None, [unit_name, unit_info_1, unit_info_2], modifier, None, 3)
        if amnt is not None and len(amnt) > 0:
            amount = int(amnt[0])
        else:
            amount = None
        new_unit = ""
        ended = False
        c3 = ""
        for word in WORD_BOUNDARY_RX.split(unit[0]):
            if word in UNIT_STOPWORDS:
                ended = True
            if ended:
                c3 += word
            else:
                new_unit += word
            if word in UNIT_SEC_STOPWORDS:
                ended = True
        c = matches.captures("c1")
        c.extend(matches.captures("c2"))
        c.append(c3)
        if new_unit!="":
            return _NameCandidate(new_unit.strip(), c, modifier, amount, 2)
        else:
            return _NameCandidate(new_unit.strip(), c, modifier, amount, 3)
    return _NameCandidate(None, [x.strip() for x in [unit_name, unit_info_1, unit_info_2] if len(x.strip()) > 0], modifier, None, 4)

def _unit_candidate(row):
    unit_amount = 1.0 if math.isnan(row["amount"]) else row["amount"]
    unit_name, _unit_comments, modifier, unit_amount_2,parser_confidence = _resolve_unit_name(str(row["name"] or ""), str(row["portion_description"] or ""), str(row["modifier"] or ""))
    singular_name = unit_name
    if unit_amount is None:
        unit_amount = unit_amount_2 if unit_amount_2 is not None else 1
    if unit_name is not None and len(unit_name) > 3 and unit_name.endswith("s"):
        plural_u_name = unit_name
        singular_name = unit_name[:-1]
    elif unit_name is not None and unit_amount == 1 and len(unit_name) > 0 and not unit_name.endswith("s"):
        plural_u_name = f"{unit_name}s"
    else:
        plural_u_name = unit_name
    if unit_amount is None or unit_amount <= 0:
        unit_amount = 1
    gram_weight = (row.gram_weight / unit_amount)
    comments_sane = []
    for comment in _unit_comments:
        splt = comment.split(",")
        for it in splt:
            rx_res = COMMENT_CLEAN_RX.match(it)
            if rx_res is None:
                continue
            g1 = rx_res.captures(1)
            comments_sane.extend([it for it in g1 if len(it) > 0])
    return {
            "unit_name": unit_name,
            "singular_name": singular_name,
            "plural_name": plural_u_name,
            "comments": comments_sane,
            "gram_weight": gram_weight,
            "entry_id": row["id_portion"],
            "measure_unit_id": row["id_measure"],
            "seq_num": int(row["seq_num"]) if row.seq_num.is_integer() else 1,
            "modifier":modifier,
            "source": f"f_{row["fdc_id"]}",
        }

def get_nutris(id: str) -> Nutris:
    rows = df_nutris.loc[df_nutris.fdc_id==id].iterrows()
    nutris = Nutris(energy=None, carbs=None, protein=None, fat=None, sat_fat=None, sugar=None, salt=None, fiber=None, calculated=[])
    for _,row in rows:
        nid = row.nutrient_id
        amnt = max(row.amount,0)
        if nid in FDC_ENERGY_IDS and nutris.energy is None:
            if nid != 1062:
                amnt *= KCAL_TO_KJ_CNST
            nutris.energy = amnt
        if nid in FDC_CARB_IDS and nutris.carbs is None:
            nutris.carbs = amnt
        if nid in FDC_FAT_IDS and nutris.fat is None:
            nutris.fat = amnt
        if nid in FDC_PROTEIN_IDS and nutris.protein is None:
            nutris.protein = amnt
        if nid in FDC_SUGAR_IDS and nutris.sugar is None:
            nutris.sugar = amnt
        if nid in FDC_FIBER_IDS and nutris.fiber is None:
            nutris.fiber = amnt
        if nid in FDC_SALT_IDS and nutris.salt is None:
            nutris.salt = amnt
        if nid in FDC_SODIUM_IDS and nutris.salt is None:
            nutris.salt = amnt * 2.5
    _fill_nutris(nutris)
    return nutris
cnt = 0
def run_it(id:str):
    global cnt
    if cnt % 100 == 0:
        print(f"Processed {cnt}")
    nutris = get_nutris(id)
    rows = df_portion_names[df_portion_names.fdc_id==id].iterrows()
    units_new = []
    for _,row in rows:
        unit_cand = _unit_candidate(row)
        units_new.append(unit_cand)
    cnt += 1
    return id, nutris.to_dict(), units_new

if __name__ == '__main__':
    nutris_json = {}
    portions_json = {}
    str_cols = ["portion_description","modifier","data_points","footnote","min_year_acquired","data_points","derivation_id","min","max","median","loq","footnote","min_year_acquired","percent_daily_value","data_type","description","publication_date","name"]
    int_cols = ["id", "fdc_id", "measure_unit_id", "seq_num", "nutrient_id", "id_measure", "id_portion"]
    dtype:DtypeArg = {x:y for x,y in ([(k,"object") for k in str_cols] + [(k,"Int64") for k in int_cols])}
    df_nutris = pandas.read_csv("./data/food_nutrient.csv", dtype=dtype).fillna(None)
    df_portion = pandas.read_csv("./data/food_portion.csv",dtype=dtype).fillna(None)
    df_measure = pandas.read_csv("./data/measure_unit.csv", dtype=dtype).fillna(None)
    df_food = pandas.read_csv("./data/food.csv",dtype=dtype).fillna(None)
    df_food_fltr:pandas.DataFrame = df_food[(df_food.data_type == "foundation_food") | (df_food.data_type == "sr_legacy_food") | (df_food.data_type == "survey_fndds_food")]

    df_portion_names = pandas.merge(df_portion, df_measure, left_on="measure_unit_id", right_on="id", how="left", suffixes=("_portion", "_measure"))
    all_ids = set([int(k) for k in df_food_fltr["fdc_id"].to_list() if k is not None and k != ""])
    print(len(all_ids))

    def fold_it(lst, it):
        lst[0][it[0]] = it[1]
        lst[1][it[0]] = it[2]
        return lst

    with ThreadPool(20) as pool:
        items = pool.map(run_it, all_ids)
        nutris_json, portions_json = reduce(fold_it, items, ({}, {}))

    with open("./data/fdc_nutris.json", "w") as f:
        json.dump(nutris_json, f, indent=4)
    with open("./data/fdc_portions.json", "w") as f:
        json.dump(portions_json, f, indent=4)
    df_food_fltr.to_csv(f"./data/all-foodbase-descriptions.csv")
