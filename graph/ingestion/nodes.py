from uuid import uuid4
import fitz

from graph.ingestion.state import IngestionState
from langchain_text_splitters import RecursiveCharacterTextSplitter
from services.chroma import chroma_service
from docx import Document

def extract_node(state: IngestionState) -> dict:
    file_path = state.get('file_path')
    text = ""

    if not file_path:
        return {"error" : "File path not provided"}

    try:
        if file_path.endswith(".pdf"):
            doc = fitz.open(file_path)
            for page in doc:
                text += page.get_text()

            doc.close()
        elif file_path.endswith(".docx"):
            doc = Document(file_path)
            text = "\n".join(p.text for p in doc.paragraphs)

        elif file_path.endswith(".txt"):
            with open(file_path, "r") as f:
                text = f.read()
        else:
            return {"error" : "File extension not provided"}
    except Exception as e:
        return {"error" : f"File not provided {str(e)}"}

    return {"raw_text" : text}

def chunk_node(state: IngestionState) -> dict:
    raw_text = state.get("raw_text")
    error = state.get("error")

    if error:
        return {}

    if not raw_text:
        return {"error" : "No text provided"}

    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size = 600,
            chunk_overlap = 100,
            length_function = lambda x: len(x),
            separators = ["\n\n","\n"," ",""]
        )
        chunks = splitter.split_text(raw_text)
    except Exception as e:
        return { "error" : str(e)}

    return {"chunks": chunks, "error" : None}

def embed_node(state: IngestionState) -> dict:
    chunks = list(state.get("chunks"))
    file_path = state.get("file_path")
    error = state.get("error")
    doc_id = str(uuid4())

    if error:
        return {}

    if not chunks:
        return {"error" : "No chunks provided"}
    try:
        result = chroma_service.add_documents(chunks, doc_id, file_path)
    except Exception as e:
        return { "error" : str(e)}
    return {"doc_id" : result["doc_id"], "error" : None}








