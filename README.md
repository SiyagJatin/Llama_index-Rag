# 🤖 RAG Agent for Research Papers

A **Retrieval-Augmented Generation (RAG) agent** built with LlamaIndex that enables intelligent question-answering and summarization over research papers. The agent uses a **ReAct (Reasoning + Acting)** loop with dual retrieval tools — vector search and summary — backed by OpenAI GPT-4o-mini and HuggingFace embeddings.

---

## 🚀 Features

- **Semantic Document Chunking** — Uses `SemanticSplitterNodeParser` with a secondary `SentenceSplitter` fallback for optimal chunk sizing
- **Dual Retrieval Strategy** — Vector-based similarity search + tree-summarized summary index
- **Page-Level Filtering** — Query specific pages of a paper using metadata filters
- **ReAct Agent with Memory** — Maintains multi-turn conversation context via `ChatMemoryBuffer`
- **Built-in Evaluation Pipeline** — Scores responses on faithfulness, answer relevancy, context relevancy, and correctness

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| Framework | [LlamaIndex](https://www.llamaindex.ai/) |
| LLM | OpenAI `gpt-4o-mini` |
| Embedding Model | `BAAI/bge-base-en-v1.5` (HuggingFace) |
| Agent Type | ReAct Agent |
| Evaluation | LlamaIndex `BatchEvalRunner` |

---

## 📁 Project Structure

```
.
├── main.ipynb          # Main notebook with full pipeline
├── .env                # API keys (not committed)
└── README.md
```

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/SiyagJatin/Llama_index-Rag.git
cd Llama_index-Rag
```

### 2. Install dependencies

```bash
pip install llama-index llama-index-embeddings-huggingface llama-index-llms-openai \
            llama-cloud python-dotenv nest_asyncio pandas
```

### 3. Configure API Keys

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_openai_api_key
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
```

### 4. Add your research paper

In `main.ipynb`, update the file path in the document loader cell:

```python
document = SimpleDirectoryReader(input_files=["path/to/your/paper.pdf"]).load_data()
```

---

## 🧠 How It Works

1. **Load & Chunk** — The PDF is loaded and split using semantic boundaries, with oversized chunks further split by sentence.
2. **Index** — Two indexes are built: a `VectorStoreIndex` for precise retrieval and a `SummaryIndex` for holistic summaries.
3. **Tools** — The agent is equipped with:
   - `vector_tool` — searches by semantic similarity, optionally filtered by page number
   - `paper_summarizer` — generates tree-summarized responses for broad questions
4. **Agent** — A ReAct agent reasons over the tools to answer queries step-by-step.
5. **Evaluate** — Responses are scored across four metrics using LlamaIndex evaluators.

---

## 💬 Usage Example

```python
response = await agent.run("What is the main contribution of this paper?")
print(response)
```

**Sample output:**
> *"The paper discusses how to enhance retrieval-augmented generation (RAG) systems by using a multi-agent reinforcement learning framework..."*

---

## 📊 Evaluation Metrics

| Metric | Description |
|---|---|
| **Faithfulness** | Are the claims grounded in the retrieved context? |
| **Answer Relevancy** | Does the response actually answer the question? |
| **Context Relevancy** | Is the retrieved context relevant to the query? |
| **Correctness** | How accurate is the answer vs. the ground truth? |

---

## 📝 Notes

- The `specific_tool` (direct vector query engine) is included as an optional third tool and can be uncommented in the tools list.
- Evaluation requires ground truth references for the correctness metric; queries without references will skip that metric.
- For unauthenticated HuggingFace requests, rate limits may apply. Set a `HF_TOKEN` for higher limits.

---

## 📄 License

MIT License
