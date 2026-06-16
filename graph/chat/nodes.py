import os

from dotenv import load_dotenv
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
    error = state["error"]

    if error:
        return {}

    if not chunks:
        return { "error": "No chunks provided" }

    system_prompt  = (
        "You are a helpful, factual, and precise assistant. Your task is to answer "
        "the user's question accurately using ONLY the provided context below.\n\n"
        "Strict Rules:\n"
        "1. Rely only on the clear facts directly mentioned in the context. Do not assume, "
        "extrapolate, or bring in outside knowledge.\n"
        "2. If the context does not contain enough information to answer the question, "
        "respond strictly with: 'I cannot find the answer based on the provided context.' "
        "Do not attempt to make up an answer.\n"
        "3. Keep your response concise, objective, and directly relevant to the query.\n\n"
    )
    generate_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt + "\n\nContext:\n{context}"),
        MessagesPlaceholder("messages"),
    ])
    context_text = "\n\n".join(chunks)
    try:
        generate_chain = generate_prompt | llm
        result = generate_chain.invoke({
            "context": context_text,
            "question": question,
            "messages": state["messages"],
        })
    except Exception as e:
        return { "error": str(e) }

    return {"messages": [AIMessage(content=result.content)], "answer": result.content}

