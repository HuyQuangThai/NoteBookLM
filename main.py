import os
import shutil
import uuid

from fastapi import FastAPI, UploadFile

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
    config = {"configurable":{"thread_id" : request.thread_id}}
    initial_state = {
        "messages": [HumanMessage(content = request.question)],
        "doc_ids": request.doc_ids
    }

    async def event_generator():
        async for event in chat_graph.astream_events(initial_state, config, version="v2"):
            if event["event"] == "on_chat_model_stream":
                chunk = event["data"]["chunk"].content
                if chunk:
                    yield f"data : {chunk}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")