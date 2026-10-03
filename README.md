# Forkulous 🍴✨

> ***WARNING:*** Forkulous is very much a work-in-progress. It is not production-ready by any means. Expect breaking changes, edge cases and missing features/data.

### What is Forkulous?

Forkulous is a free-as-in-freedom, fully local, self-hostable API for turning unstructured recipe ingredient text into nutritional information, using only local data from USDA FoodData Central and ingredient-parser-nlp, as well as various transformers for heuristic nutrient calculation. Though its main feature and endpoint is `/search`, which mainly gives nutrient information, there will also be features for food ontology (such as whether a food is derived from animals, a vegetable, etc.) in the future.

It started out as a part of foodie-rs, but quickly became big enough to be a separate application. Foodie-RS is a (as of yet unreleased) rust-based meal planner that uses an evolutionary algorithm for finding weekly meal plans. Similar to services such as EatThisMuch, only FOSS and written in Rust.

### Running Forkulous

Forkulous requires two things to run: The USDA FoodData Central dataset, in a JSON format that Forkulous likes, and the transformer models for searching it.

1. Download the [FoodDataCentral](https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_csv_2026-04-30.zip) dataset.
2. Convert the dataset using [this script](https://codeberg.org/Foodie-RS/forkulous/src/branch/main/src/scripts/generate_foodbase_json.py) (**NOTE:** This script currently uses a database connection. There is no reason it can't work from the CSV file, I just haven't found the time to change the script, but it should be fairly easy)
3. Download the models (Not yet released)
4. Prepare a virtual environment with all dependencies from requirements.txt
5. Adjust the paths in the arguments to the load2 function in [this file](https://codeberg.org/Foodie-RS/forkulous/src/branch/main/src/rest_api.py)
6. Run it using
```bash
python -m api.rest_api
```
7. Wait for the API to start, then head to https://localhost:12345/docs to test it out!
8. Profit!

### A note on architecture
Turning meatsack-produced recipes into information that a computer can understand is hard. The logic for it can be overly complex, redundant and confusing. That's why Forkulous is based on a **lazily executed pull-based pipeline**. This means:
- Functions/classes that can provide something (Forkulous calls them `Provider`s) need to be registered so that the pipeline knows to use them.
- The pipeline stores all data that has been fetched/generated during every request in a single object called `RequestState`. Notably, it also stores all `Provider`s, which MIGHT produce data in the future.
- `RequestState` is essentially a wrapper for a dictionary. Use `.get(name)` (where `name` is the name of the class you want to fetch) to access it. For instance, you can do `.get(Density)` to get the Density of the item in the current request.
- If you call `.get()` for some class that is registered, but whose `Provider` has not been executed yet, `RequestState` will automatically execute it (and, notably, cache it) under the hood, before returning to your program.
- **Subsequent calls to `.get()` for the same class on the same `RequestState` will return the cached result without re-execution of the `Provider`.**
- Forkulous does everything with the state object. For instance, the main API method is essentially just 3 lines of code:
```python
state = self.default_state.copy() # Create the `RequestState` object by copying the default state (which contains the registered `Provider`s). 
state.set(SearchRequest, req) # Store the `SearchRequest` in the newly created state.
search_res = state.get(SearchResult) # Fetch the result. This causes the `Provider` for `SearchResult` to be executed, which in turn causes the ingredient search, unit search, etc. 
```
Doing it like this has many advantages, but also some disadvantages:
- All `Provider`s are isolated from each other, making it easy to test and debug them.
- Forkulous is extremely extensible and can easily be changed to use different data sources.
- Only code that needs to run is actually executed, speeding up the request.
- `Provider`s cannot have side effects. That means that for the same state object, it should ALWAYS return the same result. 
- Stacktraces are often very long, because of all the nested execution.
- It's easy to create dependency cycles (which will result in infinite recursion).
- No guarantees can be made on if or when a `Provider` is executed. There is no explicit dependency system for providers.
- It is necessary that all `Provider`s return the types that they are supposed to return. Notably, you should never return `None`. Use `OptionalProvider` and `Option.none()` instead.

### AI
Forkulous, like Foodie-RS, is a human-based project. I consider programming an art, as much as a craft. I don't intend to use AI for most of the project. However, I did use AI on a number of occasions: to explain to me concepts that I am unfamiliar with, or to give me snippets for things that I need. Most notably, the dataset for training the semantic search of the ingredient database has been created almost entirely by an LLM, as I simply cannot annotate 10,000 tuples by myself.

I don't have a clear opinion on AI, but I know that I don't want Foodie-RS to be an AI project. I want to overcome challenges, experience the wonder of learning things I didn't know before, create something just the way I wanted it, and (hopefully) to share my creation with others and see what they make with it in turn.

That is all.
