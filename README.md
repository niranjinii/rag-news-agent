# A Checkpointed Multi-Agent RAG Pipeline for Automated Technical News Writing

This repository contains a Defensive Newsroom architecture: a stateful, multi-agent Retrieval-Augmented Generation (RAG) pipeline designed to automate the production of high-fidelity technical journalism. 

Built with LangGraph, this system shifts the paradigm from generative volume to verifiable integrity by isolating tasks into distinct, strictly contracted agentic nodes. It features a preemptive rumour kill-switch, semantic vector reranking, and a 3-layer mathematical evaluation engine to neutralize premise hallucinations and citation drift.

## System Architecture

The workflow executes across a Directed Acyclic Graph (DAG) managed by LangGraph, integrating three specialized agents and an enterprise-grade state persistence layer.

### 1. Agent 1: The Defensive Researcher (Gatekeeper & Miner)
* **Pre-Search Validation:** Evaluates live Google snippets to detect unreleased or speculative products, triggering a hard pipeline halt to prevent premise hallucinations.
* **Semantic Harvester:** Utilizes `Trafilatura` and `PyMuPDF` for web/PDF scraping, followed by a BGE Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) to rerank context.
* **JSON Trapdoor Extraction:** Deploys Llama 3.3 70B (via Groq) to extract strict JSON facts, forcing a self-correction state if user premises contradict extracted data.
* **Recursive Entity Resolution:** Pings the Wikidata Knowledge Graph to ensure technical claims map to real-world entities.

### 2. Agent 2: The Technical Writer
* **Custom Fine-Tuned Engine:** Utilizes a local Llama 3.1 8B model, fine-tuned via Unsloth on a semantically deduplicated dataset of technical journalism.
* **Quantized Edge Deployment:** Served locally via Ollama as a 4-bit GGUF for zero-latency drafting.
* **Deterministic Assembly:** Synthesizes the narrative and utilizes algorithmic fallback scripts to forcefully inject missing inline citations.

### 3. Agent 3: The Multi-Layer Evaluator
Replaces subjective LLM evaluation with a strict mathematical auditing matrix:
* **Layer 1 (Regex):** Validates that all numeric metrics (e.g., GHz, MB/s) and citation formats strictly match the source chunks.
* **Layer 2 (Vector Grounding):** Uses `SentenceTransformers` (`all-MiniLM-L6-v2`) to compute cosine similarity, flagging sentences scoring below the 0.75 threshold as hallucinations.
* **Layer 3 (LLM Disambiguation):** Routes ambiguous claims to a reasoning model for binary (YES/NO) verification.
* **Routing:** Automatically triggers a `NEEDS_REVISION` loop if accuracy or citation quality falls below policy thresholds.

---

## Installation and Setup

### Prerequisites
* Python 3.10+
* [Ollama](https://ollama.com/) (for local model inference)

### 1. Clone the Repository
```bash
git clone https://github.com/niranjinii/rag-news-agent.git
cd rag-news-agent
```

### 2. Install Dependencies
```bash
pip install -r requirements.lock.txt
```

### 3. Environment Variables
Create a `.env` file in the root directory and configure your API and local host keys:
```env
# Cloud Model APIs
GROQ_API_KEY=your_groq_key
GEMINI_API_KEY=your_gemini_key
SERPER_API_KEY=your_serper_key

# Local Model Configurations
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_API_KEY=your_ollama_key_if_hosted
```

### 4. Setup Local Inference (Ollama)
Open a new, separate terminal window and start the Ollama server:
```bash
ollama serve
```
Keep that terminal running in the background. 

**Import the Fine-Tuned Model:**
Agent 2 requires our custom fine-tuned Llama 3.1 8B model to function correctly. 
1. **Download the model:** [https://drive.google.com/file/d/18Xrz0wApQtRNI8s46XVSaFV3y0RmBD8W/view?usp=sharing]
2. Place the downloaded `.gguf` file in a known folder.
3. In that same folder, create a text file named `Modelfile` (no file extension) with the following single line:
   ```text
   FROM ./<name-of-your-downloaded-file>.gguf
   ```
4. Open a terminal in that folder and build the custom model inside Ollama by running:
   ```bash
   ollama create llama3.1 -f Modelfile
   ```
*(Note: Naming it `llama3.1` during creation ensures your existing pipeline code can call it seamlessly without needing to update the model string in the Python scripts).*
```

The terminal will prompt you to configure the exact parameters for the pipeline execution. Below is the configuration sequence:

1. **Enter search title/topic:** The technical subject to research (e.g., `NVIDIA B200 Architecture specs`).
2. **Persona:** The narrative voice for Agent 2 (Default: `Technical Journalist`).
3. **Target word count:** The integer length limit for the generated draft (Default: `800`).
4. **Enable checkpointing:** Type `Y` to enable state persistence or `N` for an ephemeral run.
5. **Checkpoint backend:** Select the storage mechanism (`memory`, `sqlite`, or `postgres`).
6. **Use unique execution ID:** Type `Y` to force a new execution ID, bypassing previous checkpoints for the same topic.
7. **Resume from existing thread_id:** Provide a specific thread ID to recover an aborted run, or leave blank to start fresh.
8. **Use injected Agent1 + Agent2 JSON files:** Type `Y` to bypass live generation and feed static mock data directly into the Evaluator (Agent 3) for deterministic testing.
   * *If Yes:* You will be prompted for the specific input JSON file paths.
9. **Show article preview / Save output JSON:** Select preferences for final output rendering and file storage.

---

## Project Structure

```text
rag-news-agent/
├── main.py                  # Interactive CLI runner for the pipeline
├── graph_pipeline.py        # LangGraph stateful DAG orchestration and routing logic
├── checkpointing.py         # Custom SQLite/Memory/Postgres state persistence manager
├── state.py                 # TypedDict definitions mapping the pipeline memory contract
├── requirements.lock.txt    # Frozen dependency tree
│
├── agents/                  # Core Agent Node Logic (Researcher, Writer, Evaluator)
├── tools/                   # External API integrations (Scrapers, ChromaDB, Search)
├── finetuning/              # Dataset curation, Unsloth configs, and test outputs
├── adapters/                # Input/Output standardization and state injection
├── research_outputs/        # Agent 1's JSON outputs from each run
├── logs/                    # System telemetry and execution traces
└── docs/                    # Technical documentation and troubleshooting guides
