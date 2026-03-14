# QUICKSTART GUIDE
## How to Run the Multi-Agent Article Writer System

---

## 📋 PREREQUISITES

### 1. Python Version
- **Required**: Python 3.9 or higher
- **Recommended**: Python 3.10 or 3.11

Check your Python version:
```bash
python --version
# or
python3 --version
```

### 2. pip (Python Package Manager)
Should come with Python. Verify:
```bash
pip --version
# or
pip3 --version
```

---

## 🚀 STEP-BY-STEP INSTALLATION

### Step 1: Extract the Archive

**On Windows:**
- Right-click `article_writer_system.zip`
- Select "Extract All..."
- Choose destination folder
- Click "Extract"

**On Mac/Linux:**
```bash
# For ZIP
unzip article_writer_system.zip

# For TAR.GZ
tar -xzf article_writer_system.tar.gz
```

### Step 2: Navigate to the Folder

```bash
cd article_writer_system
```

### Step 3: (Optional but Recommended) Create Virtual Environment

**Why?** Keeps dependencies isolated and clean.

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `langgraph` - Core orchestration framework
- `langchain` - LangChain dependencies
- `aiosqlite` - SQLite checkpoint backend
- `typing-extensions` - Type hints support

**Installation takes ~30-60 seconds.**

---

## ▶️ RUNNING THE CODE

### Option 1: Run Complete Demo (Recommended First Time)

```bash
python main.py
```

**What happens:**
1. Executes 5 complete examples
2. Shows checkpointing in action
3. Creates sample outputs
4. Demonstrates resume functionality
5. Lists all checkpoints
6. Takes ~10-15 seconds

**Expected Output:**
```
🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 
EXAMPLE 1: Normal Execution with Checkpointing
🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 🚀 

======================================================================
  ARTICLE WRITER PIPELINE - STARTING
======================================================================
Topic: NVIDIA B200 GPU release
Persona: Technical Journalist
Target Word Count: 800
Checkpointing: ENABLED ✓

[CHECKPOINT] Manager initialized with backend: sqlite
[CHECKPOINT] Using SQLite backend: checkpoints.db
[GRAPH] Pipeline graph constructed successfully
[GRAPH] Max revisions configured: 2
[GRAPH] ✓ Checkpointing ENABLED (backend: sqlite)
[GRAPH] ✓ State will be saved after each node
[GRAPH] ✓ Checkpoints stored in: checkpoints.db
...
```

### Option 2: Run Custom Pipeline (Python Script)

Create a file `my_pipeline.py`:

```python
from main import run_pipeline

# Run with your topic
state = run_pipeline(
    topic="Your Article Topic Here",
    persona="Technical Journalist",  # or "Science Writer", etc.
    word_count=800,
    enable_checkpointing=True
)

# Access results
print(f"Title: {state['draft_article']['title']}")
print(f"Status: {state['evaluation']['status']}")
```

Run it:
```bash
python my_pipeline.py
```

### Option 3: Run Custom Pipeline (Interactive Python)

```bash
python
```

Then in Python:
```python
from main import run_pipeline

# Quick run
state = run_pipeline("Quantum Computing Advances")

# Access any part of the result
print(state['draft_article']['title'])
print(state['evaluation']['scores'])
print(state['research_data']['claims'][0])
```

### Option 4: Resume from Checkpoint

If a pipeline crashes or you interrupt it:

```python
from main import run_pipeline
from checkpointing import generate_thread_id

# Get the thread ID from previous run
topic = "NVIDIA B200 GPU release"
thread_id = generate_thread_id(topic)
print(f"Thread ID: {thread_id}")  # e.g., "article_abc123"

# Resume
state = run_pipeline(
    topic=topic,
    resume_from=thread_id  # ← Resume from this checkpoint
)
```

---

## 📊 UNDERSTANDING THE OUTPUT

### Console Output Structure

```
1. PIPELINE STARTING
   ├─ Configuration summary
   └─ Checkpointing status

2. GRAPH COMPILATION
   ├─ Graph construction
   └─ Checkpoint backend info

3. EXECUTION
   ├─ [RESEARCH AGENT] Processing...
   │  └─ ✓ Checkpoint will be saved
   ├─ [WRITER AGENT] Generating...
   │  └─ ✓ Checkpoint will be saved
   ├─ [EVAL AGENT] Evaluating...
   │  └─ ✓ Checkpoint will be saved
   └─ [ROUTER] Deciding next step...

4. COMPLETION
   ├─ Final state summary
   ├─ Evaluation scores
   ├─ Article details
   └─ Checkpoint info
```

### Generated Files

After running, you'll see:

1. **checkpoints.db** - SQLite database with all checkpoints
2. **article_output.json** - Final article + metadata
3. **article_quantum_computing.json** - Second example output

---

## 🔍 EXPLORING CHECKPOINTS

### List All Saved Checkpoints

```python
from main import list_all_checkpoints

list_all_checkpoints()
```

**Output:**
```
📦 Found 2 pipeline execution(s):

1. Thread ID: article_abc123
   Latest Checkpoint: 2026-02-16T10:30:00
   Created: 2026-02-16T10:28:45

2. Thread ID: article_def456
   Latest Checkpoint: 2026-02-16T10:35:00
   Created: 2026-02-16T10:33:12
```

### Inspect Checkpoint Details

```python
from checkpointing import get_checkpoint_manager

manager = get_checkpoint_manager()
info = manager.get_checkpoint_info("article_abc123")

print(f"Thread: {info['thread_id']}")
print(f"Checkpoint ID: {info['checkpoint_id']}")
print(f"Created: {info['created_at']}")
```

### Clean Up Old Checkpoints

```python
from main import cleanup_old_checkpoints

# Delete checkpoints older than 7 days
cleanup_old_checkpoints(days=7)
```

---

## 🎮 COMMON USE CASES

### Use Case 1: Generate Article with Checkpointing

```python
from main import run_pipeline, save_results

# Generate article
state = run_pipeline(
    topic="Latest Developments in AI",
    persona="Tech Journalist",
    word_count=1000
)

# Save to file
save_results(state, "my_article.json")
```

### Use Case 2: Simulate Crash and Resume

```python
import sys
from main import run_pipeline
from checkpointing import generate_thread_id

topic = "Test Article"
thread_id = generate_thread_id(topic)

# First run - simulate crash
try:
    state = run_pipeline(topic)
    # Simulate crash: uncomment next line
    # sys.exit(1)
except:
    print(f"Crashed! But checkpoint saved: {thread_id}")

# Resume
print("Resuming from checkpoint...")
state = run_pipeline(topic, resume_from=thread_id)
print("Success!")
```

### Use Case 3: Multiple Independent Executions

```python
from main import run_pipeline

# Each gets unique thread ID
state1 = run_pipeline("Topic 1", unique_execution=True)
state2 = run_pipeline("Topic 2", unique_execution=True)
state3 = run_pipeline("Topic 3", unique_execution=True)

# All execute independently, no checkpoint conflicts
```

### Use Case 4: Disable Checkpointing (Testing)

```python
from main import run_pipeline

# Fast execution without persistence
state = run_pipeline(
    topic="Quick Test",
    enable_checkpointing=False  # No checkpoints saved
)
```

---

## 🐛 TROUBLESHOOTING

### Problem: "ModuleNotFoundError: No module named 'langgraph'"

**Solution:**
```bash
pip install -r requirements.txt
```

### Problem: "Permission denied" on Linux/Mac

**Solution:**
```bash
chmod +x main.py
python3 main.py
```

Or use virtual environment (recommended).

### Problem: "SyntaxError" when running

**Cause:** Python version too old (< 3.9)

**Solution:**
```bash
# Check version
python --version

# If < 3.9, install Python 3.10+
# Then use:
python3.10 main.py
```

### Problem: Checkpoint database is locked

**Cause:** Multiple processes accessing same checkpoint file

**Solution:**
- Stop other running pipelines
- Or use unique thread IDs:
```python
run_pipeline(topic, unique_execution=True)
```

### Problem: Import errors with agents

**Cause:** Not in correct directory

**Solution:**
```bash
# Make sure you're in the project root
cd article_writer_system
python main.py
```

---

## 📁 FILE STRUCTURE REFERENCE

```
article_writer_system/
│
├── 🟢 main.py              ← START HERE (run this first)
│   Entry point with 5 complete examples
│
├── graph_pipeline.py       ← Graph orchestration logic
│   Compiles LangGraph with checkpointing
│
├── checkpointing.py        ← Checkpoint management
│   Handles all state persistence
│
├── state.py                ← Type definitions
│   Schemas for all data structures
│
├── requirements.txt        ← Dependencies
│   Run: pip install -r requirements.txt
│
├── agents/                 ← Agent implementations
│   ├── research_dummy.py   (Agent1 - dummy)
│   ├── writer_dummy.py     (Agent2 - dummy)
│   └── eval_dummy.py       (Agent3 - dummy)
│
└── README.md               ← Full documentation
    Complete usage guide
```

---

## ⚡ QUICK REFERENCE

### Most Common Commands

```bash
# 1. Install
pip install -r requirements.txt

# 2. Run demo
python main.py

# 3. Custom run
python -c "from main import run_pipeline; run_pipeline('Your Topic')"

# 4. List checkpoints
python -c "from main import list_all_checkpoints; list_all_checkpoints()"

# 5. Cleanup
python -c "from main import cleanup_old_checkpoints; cleanup_old_checkpoints(7)"
```

### Import Reference

```python
# Run pipelines
from main import run_pipeline, save_results, display_article_preview

# Checkpoint management
from main import list_all_checkpoints, cleanup_old_checkpoints

# Direct checkpoint access
from checkpointing import (
    get_checkpoint_manager,
    generate_thread_id,
    get_thread_id_with_timestamp
)

# Graph compilation
from graph_pipeline import (
    compile_article_writer_graph,
    create_initial_state
)
```

---

## 🎯 NEXT STEPS

After running the demo:

1. **Explore the output**
   - Check `article_output.json`
   - Examine `checkpoints.db`
   - Read console logs

2. **Customize parameters**
   - Change topic
   - Adjust word count
   - Try different personas

3. **Test checkpointing**
   - Interrupt a pipeline (Ctrl+C)
   - Resume with same thread_id
   - Verify state persistence

4. **Replace dummy agents**
   - Implement real RAG in `agents/research_real.py`
   - Integrate LLM in `agents/writer_real.py`
   - Build evaluator in `agents/eval_real.py`
   - Update imports in `graph_pipeline.py`

---

## 💡 TIPS

1. **Always use virtual environment** to avoid dependency conflicts
2. **Start with main.py** to see everything working
3. **Check checkpoints.db** to understand state persistence
4. **Use unique_execution=True** when testing to avoid conflicts
5. **Keep checkpoints cleaned** with periodic cleanup_old_checkpoints()

---

## 📞 QUICK HELP

**If stuck:**
1. Check you're in correct directory: `ls` should show `main.py`
2. Verify Python version: `python --version` (need 3.9+)
3. Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`
4. Run with verbose Python: `python -v main.py`

**Still not working?**
- Check the README.md for detailed documentation
- Examine console error messages carefully
- Verify all files extracted correctly

---

## ✅ SUCCESS CHECKLIST

- [ ] Python 3.9+ installed
- [ ] Navigated to `article_writer_system/` directory
- [ ] Created virtual environment (optional but recommended)
- [ ] Ran `pip install -r requirements.txt`
- [ ] Ran `python main.py` successfully
- [ ] Saw "DEMONSTRATION COMPLETE" message
- [ ] Found `checkpoints.db` file created
- [ ] Found `article_output.json` file created

**If all checked: You're ready to go! 🎉**

---

**Happy Coding! 🚀**
