import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request

from api import ingredient_search
from api.models import SearchRequest


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    ing_api = ingredient_search.load2(
        be_model="Forkulous/be-forkulous",
        ce_model="Forkulous/ce-forkulous-large",
        index_file="../data/foodbase-embeddings.dat",
        corpus_file_csv="../data/all-foodbase-descriptions.csv",
        nutri_file="../data/fdc_nutris.json",
        portion_file="../data/fdc_portions.json",
        id_column="fdc_id",
        corpus_doc_column="description",
        unit_ranker="sentence-transformers/all-MiniLM-L6-v2",
    )
    app.state.api = ing_api
    yield
app = FastAPI(lifespan=lifespan)


@app.post("/search")
async def search(request:Request, search_req:SearchRequest):
    logger = logging.getLogger("ingr_api")
    api:ingredient_search.SearchAPI = request.app.state.api
    logger.debug("Received Search request for: %s", search_req.query)
    res = api.search_ingredient(search_req)
    logger.log(2, "Request: %s", search_req.model_dump_json(indent=4))
    logger.log(2, "Response: %s", res.model_dump_json(indent=4))
    return res

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger("ingr_api")
    logger.setLevel(level=5)
    uvicorn.run(app, host="0.0.0.0", port=12345)
