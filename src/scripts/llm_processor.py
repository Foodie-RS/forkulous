import csv
import json
import os
import random
import time

from google import genai
from google.genai import types
from pydantic import BaseModel, Field


class MatchingResult(BaseModel):
    usda: str = Field(description="The exact original USDA FoodBase string.")
    foodon: str = Field(description="The exact original FoodOn string.")
    reasoning: str = Field(description="A 1-2 sentence explanation of the decision.")
    verdict: str = Field(
        description="Must be strictly one of: match, partial match, no match, unsure."
    )


class BatchResult(BaseModel):
    results: list[MatchingResult]


INPUT_CSV = "samples_scored_inp.csv"
INPUT_JSON = "dataset_new.json"
OUTPUT_JSON = "dataset_full_lines_flash_lite_10k.json"
BATCH_SIZE = 50
MODEL_NAME = "gemini-3.1-flash-lite"

SYSTEM_INSTRUCTION = """
You are an expert at semantic matching of food items. Your task is to process a dataset for training a transformer/cross-encoder model. You will match food items from the USDA FoodBase (`usda`) to food items pulled from the FoodOn dataset (`foodon`).

Evaluation Rules:
1. Assess if both texts refer to the same fundamental food item, with regard to the culinary applications, dietary requirements and nutritional value.
2. Ignore the format of both texts (they may be a word, phrase, full sentence or an entire paragraph).
3. Ignore markdown, formatting artifacts, excess whitespace, capitalization (unless it changes the food item), or seemingly out-of-place numbers.
4. FoodOn items often contain parentheses starting with "path:" (e.g., `(path: plant root vegetable...)`). Treat these as additional context showing the location within the FoodOn graph. Parentheses *not* starting with "path:" are standard text.
5. Assume all texts are food-related (e.g., "date" refers to the fruit).
6. Ignore singular/plural differences.
7. Ignore "sizes" unless the modifier changes the fundamental food item (e.g., "large onion" -> "onion" is a Match; "bean" -> "giant stock-bean" is No Match).
8. Ignore states of preparation that do not alter the nutritional value of the product. (e.g. "diced carrot" -> "carrot" is a Match, but "rice" -> "fried rice" is a Partial Match, because cooking rice and adding oil for frying alters its nutritional value)
9. Food items that are derived from one another are a No Match, unless the derivation is implied, or the derviation does not significantly influence culinary application, dietary requirement and nutritional value.
10. A text that describes a group of ingredients that includes the item described by the other text, but also includes a significant amount of items not described by that text, it is a Partial Match. (e.g. "lettuce" -> "leaf vegetables" is a Partial Match)
11. Branded food items are a Match if and only if the fundamental food item matches, according to the other rules.
12. "NFS" stands for "not further specified".

When making a verdict of "match", you must explicitly affirm that all three central requirements (culinary application, dietary requirements and nutritional value) are fulfilled sufficiently.
Then classify each pair into exactly one of these categories: match, partial match, no match, or unsure. Default to unsure if the tuple is too ambiguous.

Examples (usda -> foodon):
* "Onions, white, raw" -> "red onion" : partial match (the difference of the onions is explicitly mentioned, so it is a partial match, because both are still onions and their nutritional value is closely related)
* "Lettuce, arugula, raw" -> "arugula leaf (path: vegetable food product: plant leaf food product: arugula food product)" : match (this is the part of the plant usually referred to when writing "arugula")
* "Salt, table" -> "A food produced by fermenting rice, barley and/or soybeans, with salt and the mold koji-kin (Aspergillus oryzae)." : no match (this is an entire dish that only contains salt as an ingredient)
* "Leavening agents, baking powder, low-sodium" -> "garlic powder (path: seasoning, powdered)" : no match (completely unrelated powders with vastly different culinary application.)
* "Spices, pepper, black" -> "Early sweet red pepper" : no match (the usda description refers to peppercorns, whereas the foodon description refers to bell peppers, which have different culinary application.)
* "Almonds, NFS" -> "Tree nuts from the plant classified under the species Amygdalus communis L. or Prunus dulcis (Mill.) D.A.Webb, commonly known as Almonds. The part consumed/analysed is not specified. When relevant, information on the part consumed/analysed has to be reported with additional facet descriptors. In case of data collections related to legislations, the default part consumed/analysed is the one defined in the applicable legislation.[https://en.wikipedia.org/wiki/Almond] and [https://www.google.co.uk/search?tbm=isch&q=Almonds]" : match (describes almonds correctly)
* "Cauliflower, raw" -> "05300 - cauliflowers and similar- (efsa foodex2) (path: agency food product type: European Union agency food product type: efsa food classification and description system for exposure assessment (efsa foodex2): 03580 - vegetables and vegetable products (efsa foodex2): 05230 - flowering brassica (efsa foodex2))" : match (describes cauliflower correctly)
* "Horseradish" -> "13640 - horseradish roots spice and similar- (efsa foodex2) (path: 12590 - spices (efsa foodex2): 13480 - root and rhizome spices (efsa foodex2))" : match (closely describes usda item. This is the food generally referred to when writing "horseradish".)
* "Basil, raw" -> "basils and mints" : partial match (FoodOn item is a broader category than usda item)
* "Olive oil" -> "12390 - olives for oil production and similar- (efsa foodex2) (path: agency food product type: European Union agency food product type: efsa food classification and description system for exposure assessment (efsa foodex2): 10200 - legumes, nuts, oilseeds and spices (efsa foodex2): 11170 - nuts, oilseeds and oilfruits (efsa foodex2): 12380 - oil fruits (efsa foodex2))" : no match (Olive oil is a heavily processed form of the olives described by the foodon description with vastly different nutritional value)
* "Margarine, NFS" -> "butter": no match (butter and margarine have similar culinary application and possibly related nutritional value, but many dietary requirements exclude butter, as it is an animal product)
* "pepper" -> "bell pepper": unsure ("pepper" is ambiguous and could refer to either bell peppers or pepper corns. Without additional context this MUST be "unsure".)
* "Lettuce, raw" -> "romaine lettuce": partial match (romaine lettuce is a narrower category than lettuce)
* "Avocado, raw" -> "A food product deriving from one or more avocados.": partial match (This foodon item includes avocados, but also any product derived from avocado. Therefore, the foodon description is too broad making it a partial match.)
* "Spices, pepper, red or cayenne" -> "The group includes dried peppers, obtained from plants of the taxonomic group Capsicum spp. and Capsicum frutescens and Capsicum anuum, such as hot pepper or Guinea spice or aleva or bird pepper or red pepper or Cayenne pepper or tabasco pepper or Paprika. The part consumed/analysed is by default unspecified. When relevant, information on the part consumed/analysed has to be reported with additional facet descriptors.[https://en.wikipedia.org/wiki/Capsicum] and [https://www.google.co.uk/search?tbm=isch&q=Peppers, dried]": partial match (This foodon description includes any dried pepper, whereas the usda description refers to specifically red/cayenne pepper, making the foodon description too broad and therefore a partial match.)
* "Carrots, raw" -> "carrot root": match (The root is the part of the plant generally referred to when writing "carrot".)
"""

def process_data():
    client = genai.Client()

    all_rows = []
    known_tuples = []
    with open(OUTPUT_JSON, mode="r", encoding="utf-8") as f:
        ds_sofar = json.load(f)
        for e in ds_sofar:
            known_tuples.append((e["query"], e["description"]))
    if INPUT_JSON:
        print(f"Lese {INPUT_JSON} ein...")
        with open(INPUT_JSON, mode="r", encoding="utf-8") as f:
            ds = json.load(f)

            cnt = 0
            # skip = 0
            for row in ds:
                cnt += 1
                doc = row["document"]
                if doc.startswith("document: "):
                    doc = doc[10:]
                clean_row = {"query": row["alt_query"], "description": doc}
                if (row["alt_query"], doc) in known_tuples:
                    continue
                all_rows.append(clean_row)
    else:
        print(f"Lese {INPUT_CSV} ein...")
        with open(INPUT_CSV, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            all_rows = []

            seen_refs = []

            cnt = 0
            # skip = 0
            for row in reader:
                # if row["ref"] in seen_refs:
                #    continue
                # skip += 1
                # if skip < 100:
                #    continue
                cnt += 1
                clean_row = {"query": row["query"], "description": row["description"]}
                seen_refs.append(row["ref"])
                if (row["query"], row["description"]) in known_tuples:
                    continue
                all_rows.append(clean_row)

    print(f"Found {len(all_rows)} rows. {len(known_tuples)} were already known")
    sample_rows = random.choices(all_rows, k=(10000 - len(known_tuples)))
    print(sample_rows[0])
    print(f"Using {len(sample_rows)} rows.")
    time.sleep(3)

    if not os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump([], f)

    for i in range(0, len(sample_rows), BATCH_SIZE):
        batch = sample_rows[i : i + BATCH_SIZE]
        print(
            f"Processing batch {i // BATCH_SIZE + 1} (rows {i} to {i + len(batch)})..."
        )

        batch_data_str = json.dumps(batch, indent=2)

        prompt = f"Here is the batch of data to process:\n\n{batch_data_str}"

        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                    response_schema=BatchResult,
                    temperature=0.0,
                ),
            )

            result_json = json.loads(response.text)

            with open(OUTPUT_JSON, "r+", encoding="utf-8") as f:
                existing_data = json.load(f)
                existing_data.extend(result_json["results"])
                f.seek(0)
                json.dump(existing_data, f, indent=4, ensure_ascii=False)
                f.truncate()

            print(f"Batch {i // BATCH_SIZE + 1} erfolgreich gespeichert.")

            time.sleep(2)

        except Exception as e:
            print(f"Error: {e}")
            print("Skipping batch...")
            time.sleep(5)


if __name__ == "__main__":
    process_data()
