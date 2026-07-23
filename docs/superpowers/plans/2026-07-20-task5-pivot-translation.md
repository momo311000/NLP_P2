# Task 5 Pivot Translation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a minimal French-to-English-to-Swedish pivot translation pipeline that reuses the existing GRU attention architecture.

**Architecture:** A preprocessing script creates aligned English-Swedish train, validation, and test files. A single Task 5 script trains FR-to-EN and EN-to-SV attention models, evaluates each component on its own references, and writes qualitative cascade predictions without claiming a direct FR-to-SV BLEU score.

**Tech Stack:** Python 3, PyTorch, existing `tasks3.py` and `task4.py` utilities, Slurm/Apptainer.

## Global Constraints

- Keep Task 5 code small and reuse existing model and metric implementations.
- Preserve aligned sentence pairs whenever a row is removed or sampled.
- Use seed 42 and the configured 70/10/20 split.
- Do not overwrite Task 3 or Task 4 outputs.
- Do not report direct French-to-Swedish BLEU without Swedish references aligned to the French source.

---

### Task 1: English-Swedish preprocessing

**Files:**
- Create: `prepare_task5_data.py`
- Create: `task5_config.json`
- Test: `tests/test_task5.py`

**Interfaces:**
- Produces: `prepare_data(config: dict) -> dict[str, int]`
- Produces: `data/processed_task5/{train,valid,test}.{en,sv}` and `preprocessing_summary.csv`

- [ ] Write tests that create a tiny aligned corpus and assert lowercase cleaning, pairwise XML/empty-row removal, deterministic sampling, and aligned split files.
- [ ] Run `python -m unittest tests.test_task5 -v` and verify import failure because `prepare_task5_data.py` does not exist.
- [ ] Implement two-pass aligned sampling using `tasks1_2.clean_sentence` and `tasks1_2.valid_pair`.
- [ ] Run the tests and verify the preprocessing cases pass.

### Task 2: Attention component training and pivot inference

**Files:**
- Create: `task5.py`
- Modify: `tests/test_task5.py`

**Interfaces:**
- Produces: `translate_text(model, src_vocab, trg_vocab, text, max_len, device) -> str`
- Produces: `build_pivot_rows(fr_en_bundle, en_sv_bundle, limit, device) -> list[list[str]]`
- Produces: `outputs/task5/component_results.csv`, `pivot_predictions.csv`, component loss histories, and model checkpoints.

- [ ] Add a test using tiny deterministic translation models to assert that pivot rows contain French source, English reference, predicted English, oracle-English Swedish output, and pivot Swedish output.
- [ ] Run the test and verify it fails because the Task 5 functions do not exist.
- [ ] Implement one reusable attention training function using the existing Task 3 loaders/metrics and Task 4 model/training loops.
- [ ] Train FR-to-EN and EN-to-SV by calling that function twice, then generate the configured number of pivot rows.
- [ ] Save only component metrics, loss histories, checkpoints, pivot predictions, and short notes.
- [ ] Run the tests and verify all cases pass.

### Task 3: Hydra launch scripts and verification

**Files:**
- Create: `prepare_task5_data.sh`
- Create: `train_task5_gpu.sh`

**Interfaces:**
- `bash prepare_task5_data.sh` prepares English-Swedish files inside the container.
- `sbatch train_task5_gpu.sh` trains both components and writes `outputs/task5`.

- [ ] Add minimal Apptainer commands using `/home/bsa04/nlp_p2` and unbuffered Python output.
- [ ] Run `python -m compileall -q prepare_task5_data.py task5.py tests`.
- [ ] Run `python -m unittest discover -s tests -v`.
- [ ] Run a tiny CPU smoke test with temporary parallel files, one epoch, and small vocabularies.
- [ ] Inspect generated CSV headers and confirm no existing output directory was overwritten.
