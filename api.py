from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List

from app.pipeline.rag_pipeline import pipeline, run_rag

app = FastAPI()

class QueryRequest(BaseModel):
    query: str


@app.post("/upload")
async def upload(files: list[UploadFile] = File(..., media_type="application/pdf")):
    pipeline.load_documents_api(files)
    return {"message": f"{len(files)} file(s) uploaded successfully"}


@app.post("/ask")
def ask(req: QueryRequest):
    result = run_rag(req.query)
    return result