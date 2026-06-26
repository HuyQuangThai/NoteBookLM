import os
import shutil
import uuid

import json
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from dto.request.AskRequest import AskRequest
from dto.request.ChatRequest import ChatRequest
from dto.request.SummaryRequest import SummaryRequest
from graph.ingestion.ingestion import build_ingestion_graph
from graph.qa.qa import build_qa_graph
from graph.summary.summary import build_summary_graph
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from graph.chat.chat import build_chat_graph
from contextlib import asynccontextmanager
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from services.chroma import chroma_service

checkpointer_path = os.path.join(os.path.dirname(__file__), "chroma_db", "chat_memory.db")
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(UPLOAD_DIR, exist_ok=True)
chat_graph = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global chat_graph
    async with AsyncSqliteSaver.from_conn_string(checkpointer_path) as checkpointer:
        chat_graph = build_chat_graph(checkpointer)
        yield

app = FastAPI(lifespan=lifespan)
origins = [
    "http://localhost:8000"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

ingestion_graph = build_ingestion_graph()
qa_graph = build_qa_graph()
summary_graph = build_summary_graph()

@app.post("/upload")
async def upload(file: UploadFile):
    temp_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{file.filename}")
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result = ingestion_graph.invoke({"file_path" : temp_path})
    if result.get("error"):
        return {"error":result["error"]}

    return {
        "doc_id": result["doc_id"],
        "chunks": len(result["chunks"]),
    }

@app.post("/ask")
async def ask(request: AskRequest):
    result = qa_graph.invoke({"question": request.question, "doc_ids": request.doc_ids})
    if result.get("error"):
        return {"error":result["error"]}
    return {
        "answer": result["answer"],
        "sources": result["sources"],
    }

@app.post("/summary")
async def summary(request: SummaryRequest):
    result = summary_graph.invoke({"doc_id": request.doc_id})
    if result.get("error"):
        return {"error":result["error"]}
    return {
        "summary": result["final_summary"],
        "insights": result["insights"],
        "tags": result["tags"],
    }

@app.post("/chat")
async def chat(request: ChatRequest):
    print("=== DOC IDS DEBUG ===", request.doc_ids)
    config = {"configurable":{"thread_id" : request.thread_id}}
    initial_state = {
        "messages": [HumanMessage(content = request.question)],
        "doc_ids": request.doc_ids
    }

    async def event_generator():
        buffer = ""
        async for event in chat_graph.astream_events(initial_state, config, version="v2"):
            if event["event"] == "on_chat_model_stream":
                chunk = event["data"]["chunk"].content
                if chunk:
                    buffer += chunk
                    if buffer.endswith((" ", ",", ".", "!", "?", "\n")):
                        yield f"data: {buffer}\n\n"
                        buffer = ""
        if buffer:
            yield f"data: {buffer}\n\n"

        last_state = await chat_graph.aget_state(config)
        citations = last_state.values.get("citations", [])
        yield f"data: {json.dumps({'type': 'citations', 'content': citations})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/list_documents")
async def list_documents():
    try:
        result = chroma_service.list_documents()
        return {
            "documents": result,
        }
    except Exception as e:
        return {"error":str(e)}

@app.delete("/delete/{doc_id}")
async def delete_document(doc_id: str):
    try:
        result = chroma_service.delete_document(doc_id)
        return {
            "status": result["status"],
            "message": result["message"],
        }
    except Exception as e:
        return {"error":str(e)}
