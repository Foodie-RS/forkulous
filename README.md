# Forkulous 🍴🌈 [^1]

> ***WARNING:*** Forkulous is very much a work-in-progress. It is not production-ready by any means. Expect breaking changes, edge cases and missing features/data.

## What is Forkulous?

Forkulous is a free-as-in-freedom, self-hostable API for turning unstructured recipe ingredient text into nutritional information, using only local data from USDA FoodData Central and ingredient-parser-nlp, as well as various transformers for heuristic nutrient calculation. Though its main feature and endpoint is `/search`, which mainly gives nutrient information, there will also be features for food ontology (such as whether a food is derived from animals, a vegetable, etc.) in the future.

It started out as a part of foodie-rs, but quickly became big enough to be a separate application. Foodie-RS is a (as of yet unreleased) rust-based meal planner that uses an evolutionary algorithm for finding weekly meal plans. Similar to services such as EatThisMuch, only FOSS and written in Rust.

Also see: [Huggingface project page](https://huggingface.co/Forkulous)

## Running Forkulous

Forkulous requires the USDA FoodData Central dataset, in a JSON format that Forkulous likes.

1. Download the [FoodDataCentral](https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_csv_2026-04-30.zip) dataset.
2. Prepare a virtual environment with all dependencies from requirements.txt. 
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
3. Create a directory called "data" in the main project folder.
4. Extract the ZIP file you just downloaded, and put `food.csv`, `food_nutrient.csv`, `food_portion.csv` and `measure_unit.csv` into the `data` directory.
5. Convert the dataset using [this script](https://codeberg.org/Foodie-RS/forkulous/src/branch/main/src/scripts/generate_foodbase_json.py):
```bash
python src/scripts/generate_foodbase_json.py
```
6. Run the API using
```bash
cd src
python -m api.rest_api # or `fastapi dev api/rest_api.py` for debug mode
```
7. Wait for the API to start, then head to http://127.0.0.1:12345/docs to test it out!
8. Profit!

## A few notes on the architecture
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
Doing it like this has many advantages, but also some disadvantages.
Advantages:
- All `Provider`s are isolated from each other, making it easy to test and debug them.
- Only code that needs to run is actually executed, speeding up the request.
- Forkulous is extremely extensible and can easily be changed to use different data sources. You don't even need to modify the codebase at all! Consider this code block:
```python
# your_program.py
api = SearchAPI(...)
api.add_provider(SearchResult, MyCoolSearchProvider(), insert=True)
```
A call to `api.search()` will now use your own custom provider for SearchResult instead of the default one, giving you complete control over the search logic! You can even use the old provider for `SearchRequest` in your new `Provider`! Look at this:
```python
class MyCoolSearchProvider(Provider[SearchResult]):
    @override
    def deferred(self):
        return True
    
    def __init__():
        super().__init__(SearchResult)

    @override
    def execute(self, state: RequestState) -> SearchResult:
        old_result = state.get(SearchResult, allow_deferred=False)
        # do something cool with old_result
        return new_result
```
Under the hood, a call to `api.search` will call `state.get(SearchResult)`. Because `allow_deferred` is `True` by default, this call will then call `MyCoolSearchProvider.execute`, which again requests `SearchResult`. But now `allow_deferred` is `False`, so the pipeline skips your provider in this call, and goes straight to the old provider. It then executes all the usual logic, and returns the `SearchResult` not to `api.search`, but to your provider. The result of your Provider will be used as the search result for the `api.search` method.

I hope you can see why I find this abstraction cool and the right fit for the task of recipe calculations.

But there are some disadvantages and things to keep in mind:
- `Provider`s should not have side effects. That means that for the same state object, it should generally return the same result. Ideally, you should only use local data and not rely on external databases that may be modified during the request.
- Stacktraces are often very long, because of all the nested execution.
- It's easy to create dependency cycles (which will result in infinite recursion). `deferred()` can mitigate this somewhat (see above).
- No guarantees can be made on if or when a `Provider` is executed. There is no explicit dependency system for providers.
- It is necessary that all `Provider`s adhere to typing. Notably, you should never return `None`. Inherit from `OptionalProvider` and return `Option.none()` instead.

## AI
Forkulous, like Foodie-RS, is a project by humans for humans. I consider programming an art, as much as a craft. I don't intend to use AI for most of the project. However, I did use AI on a number of occasions: to explain to me concepts that I am unfamiliar with, or to give me snippets for things that I need. In particular, the dataset for training the ingredient search models has been created almost entirely by an LLM, as I simply cannot annotate 10,000 tuples by myself.

I don't have a clear opinion on AI, but I know that I don't want Foodie-RS to be an AI project. I want to overcome challenges, experience the wonder of learning things I didn't know before, create something just the way I wanted it, and (hopefully) to share my creation with others and see what they make with it in turn. That is what gives this project any meaning at all to me.

That is all.

## Supporting Forkulous
Like most developers of FOSS projects, I have invested a tremendous amount of my free time and energy into this project (and Foodie-RS). If Forkulous has helped you and you want to support it, I have a [Ko-Fi page](https://ko-fi.com/juliag2)!

If you want to support Forkulous by contributing, you are more than welcome to do so! Just fork (hehe) this repository, make your changes and create a pull request!

[^1]:Pronounced fork-you-luss
