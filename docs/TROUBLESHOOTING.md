# TROUBLESHOOTING GUIDE
## Common Issues and Solutions

---

## ⚠️ Issue: "ModuleNotFoundError: No module named 'langgraph.checkpoint.sqlite'"

### Cause
The LangGraph checkpoint module structure has changed between versions.

### Solution 1: Update LangGraph (Recommended)

```bash
pip install --upgrade langgraph
```

### Solution 2: Use Memory Backend (Temporary)

If you can't upgrade, modify `main.py` to use memory backend:

```python
# Find this line in main.py:
app = compile_article_writer_graph(enable_checkpointing=True)

# Change to:
app = compile_article_writer_graph(
    enable_checkpointing=True,
    checkpoint_backend="memory"  # ← Add this
)
```

**Note**: Memory backend is NOT persistent (checkpoints lost when program exits).

### Solution 3: Install Specific Version

```bash
pip install langgraph==0.2.0 --upgrade
```

---

## ⚠️ Issue: Python 3.14 Pydantic Warning

### Warning Message
```
Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater
```

### Cause
You're using Python 3.14+ which has compatibility issues with Pydantic v1.

### Solution: Use Python 3.10-3.12 (Recommended)

**Check your Python version:**
```bash
python --version
```

**If 3.14+, install Python 3.11:**
1. Download Python 3.11 from python.org
2. Create new virtual environment:
```bash
python3.11 -m venv .venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Mac/Linux
```

3. Reinstall dependencies:
```bash
pip install -r requirements.txt
```

---

## ⚠️ Issue: "Permission denied" or "Access denied"

### On Windows
```bash
# Run PowerShell as Administrator, or:
python -m pip install -r requirements.txt
```

### On Mac/Linux
```bash
pip install --user -r requirements.txt
# or
sudo pip install -r requirements.txt
```

---

## ⚠️ Issue: Import Errors from Agents

### Error Message
```
ImportError: cannot import name 'research_agent_node'
```

### Cause
Not running from correct directory.

### Solution
```bash
# Make sure you're in the project root
cd article_writer_system
ls  # Should see main.py, state.py, etc.

# Then run:
python main.py
```

---

## ⚠️ Issue: "SyntaxError" when running

### Cause
Python version too old (< 3.9).

### Solution
```bash
# Check version
python --version

# If < 3.9, install Python 3.10+
# Then use specific Python version:
python3.10 main.py
```

---

## ⚠️ Issue: Checkpoint Database Locked

### Error Message
```
sqlite3.OperationalError: database is locked
```

### Cause
Multiple processes accessing the same checkpoint file.

### Solution 1: Stop Other Processes
Kill any other running Python processes using the same checkpoints.

### Solution 2: Use Unique Thread IDs
```python
from main import run_pipeline

# Each execution gets unique thread ID
state = run_pipeline(
    topic="Your Topic",
    unique_execution=True  # ← Add this
)
```

---

## ⚠️ Issue: "No module named 'aiosqlite'"

### Cause
aiosqlite not installed (needed for SQLite checkpointing).

### Solution
```bash
pip install aiosqlite
# or
pip install -r requirements.txt --force-reinstall
```

---

## ⚠️ Issue: Checkpoints Not Saving

### Symptom
No `checkpoints.db` file created after running.

### Diagnosis
Check console output for:
```
[WARNING] SQLite checkpoint not available, using Memory (not persistent)
```

### Solution
The code automatically falls back to Memory backend. To fix:

1. **Update LangGraph:**
```bash
pip install --upgrade langgraph
```

2. **Or explicitly use Memory backend:**
```python
# This is OK for testing (but not persistent)
app = compile_article_writer_graph(
    enable_checkpointing=True,
    checkpoint_backend="memory"
)
```

---

## ⚠️ Issue: "RuntimeError: This event loop is already running"

### Cause
Async/sync compatibility issue with Jupyter or IPython.

### Solution: Run from Command Line
```bash
# Don't run in Jupyter/IPython
# Use terminal instead:
python main.py
```

---

## 🔍 DIAGNOSTIC CHECKLIST

Run through this checklist to diagnose issues:

```bash
# 1. Check Python version (need 3.9-3.13)
python --version

# 2. Check you're in correct directory
ls  # Should see main.py

# 3. Check dependencies installed
pip list | grep langgraph
pip list | grep langchain

# 4. Try minimal import test
python -c "from langgraph.graph import StateGraph; print('OK')"

# 5. Try checkpoint import test
python -c "from langgraph.checkpoint.memory import MemorySaver; print('OK')"

# 6. Check virtual environment activated
which python  # Mac/Linux
where python  # Windows
```

---

## 💡 QUICK FIXES

### Fix 1: Clean Reinstall
```bash
# Remove virtual environment
rm -rf .venv  # Mac/Linux
rmdir /s .venv  # Windows

# Create fresh environment
python3.11 -m venv .venv
source .venv/bin/activate  # Mac/Linux
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run
python main.py
```

### Fix 2: Use Memory Backend (No Persistence)
If checkpoints don't work, disable them temporarily:

```python
# In main.py, find:
enable_checkpointing=True

# Change to:
enable_checkpointing=False
```

Or use memory backend:
```python
from graph_pipeline import compile_article_writer_graph

app = compile_article_writer_graph(
    enable_checkpointing=True,
    checkpoint_backend="memory"  # Not persistent but works
)
```

### Fix 3: Minimal Test
Create `test.py`:
```python
from graph_pipeline import compile_article_writer_graph, create_initial_state

# Disable checkpointing for testing
app = compile_article_writer_graph(enable_checkpointing=False)

initial_state = create_initial_state("Test Topic")
result = app.invoke(initial_state)

print("Success!")
print(f"Title: {result['draft_article']['title']}")
```

Run:
```bash
python test.py
```

---

## 📞 STILL STUCK?

### Collect Debug Info

Run this and share the output:

```bash
echo "=== Python Version ===" && python --version
echo "=== Installed Packages ===" && pip list | grep -E "langgraph|langchain|aiosqlite"
echo "=== Working Directory ===" && pwd
echo "=== Files Present ===" && ls -la
echo "=== Import Test ===" && python -c "from langgraph.graph import StateGraph; print('LangGraph OK')"
```

### Common Solutions Summary

| Issue | Quick Fix |
|-------|-----------|
| Checkpoint import error | `pip install --upgrade langgraph` |
| Python 3.14 warning | Use Python 3.11: `python3.11 -m venv .venv` |
| Permission denied | `pip install --user -r requirements.txt` |
| Import errors | `cd article_writer_system` (correct dir) |
| Database locked | Use `unique_execution=True` |
| No checkpoints saving | Use `checkpoint_backend="memory"` |

---

## ✅ VERIFICATION

After fixing, verify with:

```bash
python main.py
```

Should see:
- ✅ "CHECKPOINT Manager initialized"
- ✅ "Pipeline graph constructed successfully"
- ✅ "PIPELINE EXECUTION COMPLETE"
- ✅ No error messages
- ✅ `checkpoints.db` file created (if using SQLite)
- ✅ `article_output.json` file created

---

**If none of these solutions work, the code includes automatic fallbacks and should still run (just without persistent checkpoints).**
