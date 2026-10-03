import csv
import json
import os
import random
import time
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, Field


class MatchingResult(BaseModel):
    ingredient: str = Field(description="The exact original freetext ingredient.")
    usda: str = Field(description="The exact original USDA FoodBase string.")
    step1_fact_extract: str = Field(description="A description of both items.")
    step2_nutri_diet_comp: str = Field(description="Do the items differ in nutritional value or dietary requirements? If so, how and how significantly?")
    step3_prompt_rules: str = Field(description="Which specific rules from the prompt apply here?")
    verdict: Literal[
        "match","near match","mostly match", "partial match", "marginal match", "no match", "unsure", "invalid"
    ] = Field(
        description="Must be strictly one of: match, near match, mostly match, partial match, marginal match, no match, unsure, invalid."
    )


class BatchResult(BaseModel):
    results: list[MatchingResult]


INPUT_CSV = None
INPUT_JSON = "datasets/generated/dataset_1808.json"
OUTPUT_JSON = "datasets/llm-annotated/freetext2usda_newprompt-2.json"
PROMPT_FILE = "freetext_usda_prompt_finematch.txt"
BATCH_SIZE = 70
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")

with open(PROMPT_FILE, "r") as f:
    SYSTEM_INSTRUCTION = f.read()


def process_data():
    if not os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump([], f)

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    client = OpenAI()

    all_rows = []
    known_tuples = []
    with open(OUTPUT_JSON, mode="r", encoding="utf-8") as f:
        ds_sofar = json.load(f)
        for e in ds_sofar:
            known_tuples.append((e["usda"], e["ingredient"]))

    if INPUT_JSON:
        print(f"Lese {INPUT_JSON} ein...")
        with open(INPUT_JSON, mode="r", encoding="utf-8") as f:
            ds = json.load(f)

            cnt = 0
            # skip = 0
            for row in ds:
                cnt += 1
                clean_row = {"usda": row["document"], "ingredient": row["query"]}
                if (row["document"], row["query"]) in known_tuples:
                    continue
                all_rows.append(clean_row)
    else:
        print("input json must be specified")
        exit(1)
    print(f"Found {len(all_rows)} rows. {len(known_tuples)} were already known")

    sample_size = min(len(all_rows), max(0, 10000 - len(known_tuples)))
    if sample_size == 0:
        print("No new rows to process.")
        return

    sample_rows = random.sample(all_rows, k=sample_size)
    print(sample_rows[0:5])
    print(f"Nutze {len(sample_rows)} Zeilen.")
    time.sleep(20)

    for i in range(0, len(sample_rows), BATCH_SIZE):
        batch = sample_rows[i : i + BATCH_SIZE]
        print(
            f"Processing batch {i // BATCH_SIZE + 1} (Lines {i} to {i + len(batch)})..."
        )

        batch_data_str = json.dumps(batch, indent=2)
        prompt = f"Here is the batch of data to process:\n\n{batch_data_str}"

        try:
            response = client.beta.chat.completions.parse(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_INSTRUCTION},
                    {"role": "user", "content": prompt},
                ],
                response_format=BatchResult,
            )

            message = response.choices[0].message
            if getattr(message, "refusal", None):
                raise ValueError(f"Model refused request: {message.refusal}")
            if message.parsed is None:
                raise ValueError("OpenAI returned no structured result.")

            result_json = message.parsed.model_dump()

            with open(OUTPUT_JSON, "r+", encoding="utf-8") as f:
                existing_data = json.load(f)
                existing_data.extend(result_json["results"])
                f.seek(0)
                json.dump(existing_data, f, indent=4, ensure_ascii=False)
                f.truncate()

            print(f"Batch {i // BATCH_SIZE + 1} saved.")
            time.sleep(2)

        except Exception as e:
            print(f"Error: {e}")
            print("Skipping batch...")
            time.sleep(5)


if __name__ == "__main__":
    process_data()
