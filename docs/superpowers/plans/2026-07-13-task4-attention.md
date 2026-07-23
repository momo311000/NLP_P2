# Task 4 Attention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a separate Task 4 script that trains an RNN encoder-decoder with attention and compares it with the Task 3 baseline.

**Architecture:** Keep `tasks3.py` unchanged. Create `task4.py` that imports Task 3 utilities, adds an encoder that returns all source states, a dot/general attention module, an attention decoder, an attention seq2seq wrapper, prediction helpers, heatmap export, and a Task 4 main flow.

**Tech Stack:** Python 3, PyTorch, matplotlib, CSV output.

## Global Constraints

- Do not modify `tasks3.py`.
- Write Task 4 outputs under `outputs/task4`.
- Reuse `data/processed` train/validation/test splits.
- Compare `word_en_fr_attention` with `outputs/task3/task3_results.csv` when available.

---

### Task 1: Attention Model Test

**Files:**
- Create: `tests/test_task4_attention.py`
- Create: `task4.py`

**Interfaces:**
- Consumes: Task 3 constants and PyTorch.
- Produces: `AttentionEncoder`, `AttentionDecoder`, `Seq2SeqAttention`.

- [ ] **Step 1: Write failing test**

Create a unittest that imports `task4` and checks attention output shapes, normalization, and masking.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_task4_attention -v`
Expected: FAIL because `task4` does not exist yet.

- [ ] **Step 3: Implement Task 4 model and workflow**

Create `task4.py` with attention model classes, training/evaluation flow, heatmap output, and comparison CSV.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_task4_attention -v`
Expected: PASS.

- [ ] **Step 5: Compile check**

Run: `python -m compileall -q task4.py tests`
Expected: no output and exit code 0.
