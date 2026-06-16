# NotebookLM Agent — Project Progress

## Stack
- **Backend:** Python, FastAPI, LangGraph
- **LLM:** LLM-Stats (ChatOpenAI-compatible)
- **Embeddings:** Ollama `nomic-embed-text`
- **Vector DB:** ChromaDB (persistent local) via `langchain-chroma`
- **Env:** `.env` với `GOOGLE_API_KEY`, `LLM_STATS_API_KEY`,

## Project Structure
```
NotebookLM/
├── main.py
├── .env
├── .gitignore
├── chroma_db/              # ChromaDB + SQLite persistent storage
│   └── chat_memory.db      # conversation history (AsyncSqliteSaver)
├── data/                   # uploaded files
├── graph/
│   ├── ingestion/
│   │   ├── state.py        # IngestionState
│   │   ├── nodes.py        # extract_node, chunk_node, embed_node
│   │   └── ingestion.py    # build_ingestion_graph()
│   ├── qa/
│   │   ├── state.py        # QAState, GradeAnswer
│   │   ├── nodes.py        # retrieve_node, grade_node, generate_node
│   │   └── qa.py           # build_qa_graph()
│   ├── summary/
│   │   ├── state.py        # SummaryState, MapState
│   │   ├── nodes.py        # fetch_chunks, generate_batches, map_summary, reduce_summary, insights, tags
│   │   └── summary.py      # build_summary_graph()
│   └── chat/
│       ├── state.py        # ChatState
│       ├── nodes.py        # chat_retrieve_node, chat_generate_node
│       └── chat.py         # build_chat_graph(checkpointer)
├── dto/
│   └── request/
│       ├── AskRequest.py
│       ├── SummaryRequest.py
│       └── ChatRequest.py
└── services/
    └── chroma.py           # ChromaService
```

---

## Day 1 ✅ — Document Ingestion Pipeline

### LangGraph concepts learned
- `StateGraph` + `TypedDict` State
- Node là pure function: nhận state, trả về dict
- `add_node`, `add_edge`, `set_entry_point`, `compile()`
- LangGraph tự merge dict trả về vào state chung

### Graph flow
```
[extract_node] → [chunk_node] → [embed_node] → END
```

### Nodes
| Node | Input (từ State) | Output (vào State) |
|------|---|----|
| `extract_node` | `file_path` | `raw_text` |
| `chunk_node` | `raw_text` | `chunks` |
| `embed_node` | `chunks`, `file_path` | `doc_id` |

### IngestionState
```python
class IngestionState(TypedDict):
    file_path: str
    raw_text: str
    chunks: List[str]
    doc_id: str
    error: Optional[str]
```

### ChromaService
- Collection: `notebooklm_docs`
- Metadata mỗi chunk: `{"source": file_path, "doc_id": doc_id, "chunk_index": i}`
- IDs format: `{doc_id}_chunk_{i}`
- Embedding: Google `text-embedding-004` (dimension: 768)

### API
- `POST /upload` — nhận `UploadFile`, lưu vào `data/`, invoke graph
- Response: `{"doc_id": "...", "chunks": 174}`

### Files uploaded cho lưu vào `data/` với prefix UUID:
```
data/{uuid}_{original_filename}
```

---

## Day 2 ✅ — RAG Q&A + Conditional Edges

### LangGraph concepts learned
- `add_conditional_edges` — route theo kết quả node
- Router function trả về string → lookup trong map
- Node nên check `state.get("error")` trước khi xử lý

### Graph flow
```
[retrieve_node] → [grade_node] →(yes)→ [generate_node] → END
                              └─(no)─→ END
```

### Nodes
| Node | Input | Output |
|------|-------|--------|
| `retrieve_node` | `question`, `doc_ids` | `chunks`, `sources` |
| `grade_node` | `question`, `chunks` | `is_relevant` |
| `generate_node` | `question`, `chunks` | `answer` |

### QAState
```python
class QAState(TypedDict):
    question: str
    doc_ids: list[str]
    chunks: list[str]
    is_relevant: bool
    answer: str
    sources: list[dict]
    error: Optional[str]
```

### Structured Output
```python
class GradeAnswer(BaseModel):
    binary_score: Literal["yes", "no"] = Field(
        description="Do the chunks contain enough information to answer the question?"
    )
```

### ChromaService — methods
```python
def add_documents(self, chunks: list[str], doc_id: str, file_path: str) -> str
def query(self, question: str, doc_ids: list[str] = None, n_results: int = 5) -> list[Document]
```

### API
- `POST /ask` — nhận `AskRequest(question, doc_ids)`
- Response: `{"answer": "...", "sources": [{"doc_id", "source", "chunk_index"}]}`

### Gotchas gặp phải
- Gemini không chấp nhận system-only prompt → phải có `("human", ...)` message
- ChromaDB filter phải dùng `doc_id` trong metadata, không phải `source`
- `with_structured_output` cần test kỹ với OpenRouter proxy

---

## Day 3 ✅ — Auto-summary + Parallel Nodes

### LangGraph concepts learned
- `Send` API — fan-out chạy nodes song song
- `Annotated[list, add]` reducer — merge kết quả từ parallel nodes
- Router function trả về `list[Send]` thay vì string để fan-out

### Graph flow
```
[fetch_chunks_node]
        ↓ fan_out via Send (3 nhánh song song)
[generate_batches] [insights] [tags]
        ↓
[map_summary_node] × N     ← fan-out tiếp theo batch
        ↓ fan_in (Annotated reducer)
[reduce_summary_node]
        ↓
       END
```

### Nodes
| Node | Input | Output |
|------|-------|--------|
| `fetch_chunks_node` | `doc_id` | `chunks` |
| `generate_batches_node` | `chunks` | `batches`, `summaries=[]` |
| `map_summary_node` | `batch` (MapState) | `summaries` (append via reducer) |
| `reduce_summary_node` | `summaries` | `final_summary` |
| `insights_node` | `chunks` | `insights` |
| `tags_node` | `chunks` | `tags` |

### States
```python
class MapState(TypedDict):
    batch: List[str]

class SummaryState(TypedDict):
    doc_id: str
    chunks: List[str]
    batches: List[List[str]]
    summaries: Annotated[list[str], add]   # reducer merge parallel results
    final_summary: str
    insights: str
    tags: str
    error: Optional[str]
```

### ChromaService — method thêm
```python
def get_all_chunks(self, doc_id: str) -> list[str]:
    # dùng vector_store.get() với where={"doc_id": doc_id}
    # không cần embedding — raw get
```

### API
- `POST /summarize` — nhận `SummaryRequest(doc_id)`
- Response: `{"summary": "...", "insights": "...", "tags": "..."}`

### Gotchas gặp phải
- Router fan-out phải trả về `list[Send]` — không phải string
- `generate_batches_node` phải init `summaries: []` để reducer không bị lỗi lần đầu
- `insights_node` và `tags_node` dùng sampling (`_sample_chunks`) thay vì toàn bộ chunks để tránh context quá dài

---

## Day 4 ✅ — Multi-doc Chat + Persistent Memory + SSE Streaming

### LangGraph concepts learned
- `add_messages` reducer — tự động append message vào history thay vì overwrite
- `MessagesPlaceholder` — inject toàn bộ conversation history vào prompt
- `AsyncSqliteSaver` — persistent memory theo `thread_id`, dùng `async with` context manager
- `astream_events` — stream từng token từ LLM về client
- FastAPI `lifespan` — khởi tạo async resource (checkpointer) đúng cách

### Graph flow
```
[chat_retrieve_node] →(ok)→ [chat_generate_node] → END
                   └─(err)─→ END
```

### Nodes
| Node | Input | Output |
|------|-------|--------|
| `chat_retrieve_node` | `messages[-1].content`, `doc_ids` | `chunks`, `sources` |
| `chat_generate_node` | `messages`, `chunks` | `messages` (AIMessage appended), `answer` |

### ChatState
```python
class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]  # key: add_messages reducer
    doc_ids: List[str]
    chunks: List[str]
    sources: List[dict]
    answer: str
    error: Optional[str]
```

### Persistent Memory — AsyncSqliteSaver
```python
# chat.py — nhận checkpointer từ ngoài vào
def build_chat_graph(checkpointer):
    return graph.compile(checkpointer=checkpointer)

# main.py — dùng lifespan để giữ async context manager sống
@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        app.state.chat_graph = build_chat_graph(checkpointer)
        yield

app = FastAPI(lifespan=lifespan)
```

File DB lưu tại: `chroma_db/chat_memory.db`

### SSE Streaming
```python
async def event_generator():
    async for event in chat_graph.astream_events(state, config, version="v2"):
        if event["event"] == "on_chat_model_stream":
            chunk = event["data"]["chunk"].content
            if chunk:
                yield f"data: {chunk}\n\n"
    yield "data: [DONE]\n\n"

return StreamingResponse(event_generator(), media_type="text/event-stream")
```

### API
- `POST /chat` — nhận `ChatRequest(question, doc_ids, thread_id)`
- Response: SSE stream, kết thúc bằng `data: [DONE]`
- Cùng `thread_id` → LLM nhớ conversation history từ SQLite

### Gotchas gặp phải
- `SqliteSaver` sync không dùng được với `astream_events` → phải dùng `AsyncSqliteSaver`
- `AsyncSqliteSaver.from_conn_string()` trả về context manager, không phải object trực tiếp → phải dùng `async with` trong `lifespan`
- Path `chat.py` nằm sâu 2 cấp (`graph/chat/`) → cần 3 lần `os.path.dirname(__file__)` để lên root
- `sqlite3.connect()` không tự tạo thư mục → cần `os.makedirs(..., exist_ok=True)` trước

---

## Remaining Days

### Day 5 — Frontend + Deploy
- Simple React chat UI
- API key auth
- Docker Compose: FastAPI + ChromaDB
- Deploy Railway / Render