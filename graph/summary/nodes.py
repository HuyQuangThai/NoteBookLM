import os
from typing import List

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from graph.summary.state import SummaryState, MapState
from services.chroma import chroma_service

load_dotenv()
llm = ChatOpenAI(
    model="gpt-oss-20b",
    openai_api_key=os.getenv("LLMSTATS_API_KEY"),
    openai_api_base="https://gateway.llm-stats.com/v1",
)


def fetch_chunks_node(state: SummaryState) -> dict:
    doc_id = state.get("doc_id")
    if not doc_id:
        return {"error": "No document provided"}

    try:
        chunks = chroma_service.get_all_chunks(doc_id)
        if not chunks:
            return {"error": "No chunks found"}
    except Exception as e:
        return {"error": str(e)}
    return {"chunks": chunks}

def generate_batches_node(state: SummaryState):
    chunks = state.get("chunks")
    batch_size = 10
    batches = [chunks[i: i + batch_size] for i in range(0, len(chunks), batch_size)]
    return {"batches": batches, "summaries": []}

def map_summary_node(state: MapState) -> dict:
    batch_text = "\n\n".join(state.get("batch"))
    system_prompt = (
        "You are an AI specialized in technical document extraction. "
        "Analyze the following text and extract ONLY the core concepts, key breakthroughs, or main results. "
        "STRICTLY ignore numerical configurations, detailed round-by-round descriptions, and boilerplate text. "
        "Provide the output as brief bullet points (maximum 3 points)."
    )
    user_prompt = (
        "Text content:\n{text}"
    )

    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", user_prompt)
    ])
    try:
        summary_chain = summary_prompt | llm
        result = summary_chain.invoke({"text": batch_text})
    except Exception as e:
        return {"summarise": None, "error" : str(e)}

    return {"summaries": [result.content]}

def reduce_summary_node(state: SummaryState) -> dict:
    error = state.get("error")

    if error:
        return {}

    batch_summary = state.get("summaries")
    summaries = "\n\n".join(batch_summary)
    system_prompt = (
        "You are an expert technical editor. Synthesize the provided points into a high-level executive summary. "
        "Structure your response into 3 clear, distinct sections: \n"
        "1. Core Innovation (What is it?)\n"
        "2. Key Mechanism (How does it work? - in 1-2 sentences)\n"
        "3. Ultimate Impact (Why does it matter / What did it achieve?)\n"
        "Keep the entire output under 150 words. Avoid dense jargon and technical tables."
    )
    user_prompt = (
        "Sub-summaries:\n{summaries}"
    )
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", user_prompt)
    ])
    try:
        summary_chain = summary_prompt | llm
        result = summary_chain.invoke({"summaries": summaries})
    except Exception as e:
        return {"summarise": None, "error" : str(e)}

    return {"final_summary" : result.content}

def insights_node(state: SummaryState) -> dict:
    chunks = state.get("chunks")
    sample = _sample_chunks(chunks, max_chunks=15)
    text = "\n\n".join(sample)
    system_prompt = (
        "You are a sharp technology analyst. Based on the text, extract exactly 3-5 critical insights. "
        "Rules for each insight:\n"
        "- Must be a single, concise sentence only.\n"
        "- Start with a bolded core concept or breakthrough (e.g., **Architecture Optimization:** ... or **Performance Scalability:** ...).\n"
        "- Focus heavily on 'Why it matters', the core innovative leap, or the ultimate impact, NOT on setup statistics or background context.\n"
        "- Keep the tone professional, punchy, and easily scannable."
    )
    user_prompt = (
        "Document Content:"
        "\n\n"
        "{text}"
        "\n\n"
        "Analyze the document above and generate the insights strictly following the system rules."
    )
    insights_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", user_prompt)
    ])
    try:
        insight_chain = insights_prompt | llm
        result = insight_chain.invoke({"text": text})
    except Exception as e:
        return {"insights": None, "error" : str(e)}

    return {"insights" : result.content}

def tags_node(state: SummaryState) -> dict:
    chunks = state.get("chunks")
    sample = _sample_chunks(chunks, max_chunks=15)
    text = "\n\n".join(sample)
    system_prompt = (
        "Generate a list of metadata tags for the provided text. "
        "Rules:\n"
        "- Return MAXIMUM 5 tags.\n"
        "- Tags must be comma-separated.\n"
        "- Focus ONLY on high-level domains, core architectures, or primary methodologies "
        "(e.g., 'Machine Learning', 'Neural Networks', 'Natural Language Processing', 'Framework Design').\n"
        "- Do NOT include specific metrics, scores, versions, dates, or generic words like 'Comparison', 'Analysis', or 'Overview'."
    )
    user_prompt = (
        "Document Content:"
        "\n\n"
        "{text}"
        "\n\n"
        "Analyze the document above and generate the tags strictly following the system rules."
    )
    tags_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", user_prompt)
    ])
    try:
        tag_chain = tags_prompt | llm
        result = tag_chain.invoke({"text": text})
    except Exception as e:
        return {"tags": None, "error" : str(e)}

    return {"tags" : result.content}

def _sample_chunks(chunks: List[str], max_chunks: int) -> List[str]:
    if len(chunks) > max_chunks or max_chunks <= 1:
        return chunks[:max_chunks]
    indices = [int(i * (len(chunks) - 1) / (max_chunks - 1)) for i in range(max_chunks)]
    return [chunks[i] for i in indices]



