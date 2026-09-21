# Predictor Warmup Implementation Plan

> Execute sequentially in the main task using executing-plans; project instructions prohibit subagents.

**Goal:** Colab 07 trains the approved text-only KLUE BERT Style Predictor and resumes safely.
**Architecture:** Validated prepared manifests -> pinned tokenizer/BERT -> emotion and continuous heads -> masked losses -> update-boundary checkpoint/evaluation/export.
**Tech Stack:** PyTorch, Transformers, stdlib ZIP/JSON, existing dataset storage.
**Spec:** docs/superpowers/specs/2026-09-21-predictor-warmup-design.md (approved).

## Constraints and rulings

- Never include character IDs/audio in predictor input. No TTS training or invented context.
- Use existing split and train-only standardization unchanged.
- Initial hyperparameters are configurable experiments, not established optimum values.
- Full config/input/revision identity required for resume. A new experiment uses a new output folder.
- At a changed epoch budget/batch/LR, use a new run; no silent scheduler mutation on resume.
- Windows local fixtures prove mechanics, not Colab GPU fit or real training quality.
- No git repository / no commit or publication actions.

## Tasks

- [x] 1. Test and implement prepared-data validation, text-only model and masked soft-CE/SmoothL1.
  Files: training/predictor.py, tests/test_predictor.py, config/predictor.json, pyproject.toml.
- [x] 2. Test and implement deterministic accumulation, validation, checkpoints and resume.
  Files: training/predictor.py, tests/test_predictor.py. Test dropout RNG and partial batches.
- [x] 3. Add Colab 07 bootstrap and inference/evaluation exports, docs and notebook parity.
  Files: scripts/colab_train_predictor.py, notebooks/07_train_predictor.ipynb, docs/predictor-training.md.
- [x] 4. Run full local suite and actual small BERT smoke,
  inspect implementation for checkpoint correctness, validate bundle, report exact limits.

Completed: 67 tests passed. Also verified pinned full pretrained KLUE BERT CPU gradients on
two real samples and all prepared-data token lengths. Full T4 FP16 training remains unverified.
