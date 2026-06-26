import os
import re

from dotenv import load_dotenv
from google.ai.generativelanguage_v1alpha.types import citation
from google.generativeai import answer
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from graph.chat.state import ChatState
from services.chroma import chroma_service

load_dotenv()
llm = ChatOpenAI(
    model="gpt-oss-20b",
    openai_api_key=os.getenv("LLMSTATS_API_KEY"),
    openai_api_base="https://gateway.llm-stats.com/v1",
)

def chat_retrieve_node(state:ChatState) -> dict:
    question = state["messages"][-1].content
    doc_ids = state["doc_ids"]

    if not question:
        return {"error": "No question provided"}

    if not doc_ids:
        return { "error": "No documents provided" }

    try:
        result = chroma_service.query(question, doc_ids)
    except Exception as e:
        return { "error": str(e) }

    return {"chunks": result["chunks"], "sources": result["sources"], "error" : None}

def chat_generate_node(state:ChatState) -> dict:
    question = state["messages"][-1].content
    chunks = state["chunks"]
    sources_metadata = state["sources"]
    error = state["error"]

    if error:
        return {}

    if not chunks:
        return { "error": "No chunks provided" }

    system_prompt  = (
        "You are a precise assistant. Answer the user's question using ONLY the provided context below.\n\n"
        "Context format: [1] text..., [2] text...\n\n"
        "Rules:\n"
        "1. Use only facts explicitly stated in the context. No assumptions or outside knowledge.\n"
        "2. If the context lacks sufficient information, respond exactly: 'I cannot find the answer based on the provided context.'\n"
        "3. Be concise and relevant.\n"
        "4. Citation rules:\n"
        "   - Cite every fact with its exact chunk index in brackets (e.g., [1], [2]) at the end of the sentence.\n"
        "   - Cite ALL chunks that contributed to your answer. Do not skip any.\n"
        "   - Do not combine citations (use [1][2], not [1-2]).\n"
        "   - Do not invent citation numbers. Use only the provided chunk indices.\n"
        "   - CRUCIAL: You MUST use standard square brackets for citations, like [1], [2], [3]. NEVER use double brackets like 【1】 or 【2】 under any circumstances.\n\n"
    )
    generate_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt + "\n\nContext:\n{context}"),
        MessagesPlaceholder("messages"),
    ])
    numbered_chunks = [f"[{i+1}] {chunk}" for i, chunk in enumerate(chunks)]
    context_text = "\n\n".join(numbered_chunks)
    try:
        generate_chain = generate_prompt | llm
        result = generate_chain.invoke({
            "context": context_text,
            "question": question,
            "messages": state["messages"],
        })

        answer_text = result.content
        found_indices = re.findall(r'\[(\d+)\]', answer_text)
        unique_indices = sorted(list(set(int(idx) for idx in found_indices)))

        citations = []
        for idx in unique_indices:
            array_idx = idx - 1
            if 0 <= array_idx <= len(chunks):
                metadata = sources_metadata[array_idx] if array_idx < len(sources_metadata) else {}
                citations.append({
                    "index": idx,
                    "source": os.path.basename(metadata.get("source", "Unknown")),
                    "doc_id": metadata.get("doc_id", ""),
                    "chunk_index": metadata.get("chunk_index", None), # Tiện tay lấy luôn index gốc của chunk nếu FE cần
                    "text_snippet": chunks[array_idx]
                })

    except Exception as e:
        return { "error": str(e) }

    return {
        "messages": [AIMessage(content=result.content)],
        "answer": result.content,
        "citations": citations
    }

