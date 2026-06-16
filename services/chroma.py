from typing import List

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import os
from dotenv import load_dotenv

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
        metadatas = [
            {
                "source": file_path,
                "doc_id": doc_id,
                "chunk_index": i
            }
            for i in range(len(chunks))
        ]
        documents = [
            Document(page_content=chunks[i], metadata = metadatas[i]) for i in range(len(chunks))
        ]

        self.vector_store.add_documents(documents=documents, ids=ids)
        return doc_id

    def query(self, question: str, doc_ids: list[str] = None, n_results: int = 5):
        filter = {"doc_id":{"$in":doc_ids} if doc_ids else None}
        results = self.vector_store.similarity_search(
            query=question,
            k=n_results,
            filter=filter,
        )
        chunks = [doc.page_content for doc in results]
        sources = [doc.metadata for doc in results]
        return {"chunks": chunks, "sources": sources, "error": None}

    def get_all_chunks(self, doc_id: str) -> List[str]:
        results = self.vector_store.get(
            where={"doc_id": doc_id}
        )
        return results["documents"]

chroma_service = ChromaService()

