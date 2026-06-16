import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from graph.qa.state import QAState, GradeAnswer
from services.chroma import chroma_service


load_dotenv()
llm = ChatOpenAI(
    model="gpt-oss-20b",
    openai_api_key=os.getenv("LLMSTATS_API_KEY"),
    openai_api_base="https://gateway.llm-stats.com/v1",
)

def retrieve_node(state: QAState) -> dict:
    question = state.get("question")
    doc_ids = state.get("doc_ids")

    if not question:
        return {"error" : "No question provided"}

    if not doc_ids:
        return {"error" : "No documents provided"}

    try:
        result = chroma_service.query(question, doc_ids)
    except Exception as e:
        return { "error" : str(e)}

    return {"chunks": result["chunks"], "sources": result["sources"], "error" : None}

def grade_node(state: QAState) -> dict:
    question = state.get("question")
    chunks = state.get("chunks")

    if state.get("error"):
        return {}

    structured_llm = llm.with_structured_output(GradeAnswer)
    system_prompt = (
        "You are an expert grader assessing relevance of a retrieved context to a user question.\n"
        "Base your evaluation strictly on the provided context without making up any outside information.\n"
        "Return 'yes' if the context is sufficient, 'no' if not."
    )
    user_prompt = (
        "Context:\n{context}\n\nUser Question: {question}"
    )
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human",  user_prompt),
    ])
    context_text = "\n\n".join(chunks)
    try:
        grader_chain = grade_prompt | structured_llm
        result: GradeAnswer = grader_chain.invoke({
            "context": context_text,
            "question": question,
        })
    except Exception as e:
        return {"is_relevant" : False, "error" : str(e)}

    is_relevant = result.binary_score == "yes"
    return {
            "is_relevant" : is_relevant,
            "answer": "I don't have enough information to answer your question based on the retrieved context."
    }

def generate_node(state: QAState):
    question = state.get("question")
    chunks = state.get("chunks")

    if state.get("error"):
        return {}

    system_prompt = (
        "You are a strict QA grader checking if the retrieved context is sufficient to answer a user question.\n"
        "Carefully analyze the given text chunks (context) and the user question.\n"
        "Your core duty is to determine if the context contains enough facts to provide a complete answer.\n"
        "Do not extrapolate, assume, or bring in external knowledge. Evaluate strictly based on what is explicitly written.\n\n"
    )
    user_prompt = (
        "Context:\n{context}\n\nUser Question: {question}"
    )
    generate_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", user_prompt),
    ])
    context_text = "\n\n".join(chunks)
    try:
        generate_chain = generate_prompt | llm
        result = generate_chain.invoke({
            "context": context_text,
            "question": question,
        })
    except Exception as e:
        return {"answer": None, "error" : str(e)}

    return {"answer": result.content, "error" : None}