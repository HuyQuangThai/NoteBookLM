from typing import List, Dict

from google.generativeai.types import file_types
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import os
from dotenv import load_dotenv
from datetime import datetime, timezone

load_dotenv()

class ChromaService:
    def __init__(self):
        self.db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
        self.embedding_fn = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://localhost:11434"
        )
        self.vector_store = Chroma(
            collection_name="notebooklm_docs",
            embedding_function=self.embedding_fn,
            persist_directory=self.db_path,
        )
        self.collection = self.vector_store._collection

    def add_documents(self, chunks: list[str], doc_id: str, file_path: str):
        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        current_time = datetime.now(timezone.utc).isoformat()
        total_word_count = sum(len(chunk.split()) for chunk in chunks)
        file_type = file_path.split('.')[-1].lower() if '.' in file_path else "txt"

        metadatas = [
            {
                "source": os.path.basename(file_path),
                "doc_id": doc_id,
                "chunk_index": i,
                "uploaded_at": current_time,
                "word_count": total_word_count,
                "file_type": file_type,
            }
            for i in range(len(chunks))
        ]
        documents = [
            Document(page_content=chunks[i], metadata = metadatas[i]) for i in range(len(chunks))
        ]

        self.vector_store.add_documents(documents=documents, ids=ids)
        return {
        "doc_id": doc_id,
        "source": os.path.basename(file_path),
        "word_count": total_word_count,
        "uploaded_at": current_time
        }

    def query(self, question: str, doc_ids: list[str] = None, n_results: int = 5):
        filter = {"doc_id":{"$in":doc_ids} if doc_ids else None}
        results = self.vector_store.similarity_search(
            query=question,
            k=n_results,
            filter=filter,
        )

        print("=== QUERY RESULTS ===")
        for doc in results:
            print("doc_id:", doc.metadata.get("doc_id"))
            print("source:", doc.metadata.get("source"))
            print("---")

        chunks = [doc.page_content for doc in results]
        sources = [doc.metadata for doc in results]
        return {"chunks": chunks, "sources": sources, "error": None}

    def get_all_chunks(self, doc_id: str) -> List[str]:
        results = self.vector_store.get(
            where={"doc_id": doc_id}
        )
        return results["documents"]

    def list_documents(self):
        chunks = self.collection.get()
        metadata_list = chunks.get("metadatas", [])

        results = {}
        for metadata in metadata_list:
            if not metadata:
                continue

            doc_id = metadata.get("doc_id")
            source = metadata.get("source")
            uploaded_at = metadata.get("uploaded_at", "Unknown")
            word_count = metadata.get("word_count", 0)
            file_type = metadata.get("file_types", 'pdf')

            if not doc_id or not isinstance(doc_id, str):
                continue

            if doc_id not in results:
                results[doc_id] = {
                    "source": os.path.basename(source),
                    "uploaded_at": uploaded_at,
                    "word_count": word_count,
                    "file_type": file_type,
                }
        return results

    def delete_document(self, doc_id: str):
        self.vector_store.delete(where={"doc_id": doc_id})
        return {"status": "success", "message": f"Deleted document {doc_id}"}


chroma_service = ChromaService()

