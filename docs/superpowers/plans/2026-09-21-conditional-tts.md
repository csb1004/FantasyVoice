# Conditional TTS Implementation Plan

> Use superpowers:executing-plans sequentially in the main session, per workspace instructions.

**Goal:** Reuse Predictor v1 and train/generate Korean character speech with differentiable style conditions.
**Architecture:** A bounded style adapter feeds Melo's existing g routes. An explicit warmup/joint engine owns all optimizer updates and durable checkpoint boundaries. Cached original-audio derivatives and text features retain source identities.
**Tech Stack:** PyTorch, pinned MeloTTS, Transformers, soundfile, scipy, Colab.
**Spec:** ../specs/2026-09-21-conditional-tts-design.md (approved option 1).

## Global constraints

- Preserve 07 state dict and raw outputs; no raw audio edits, resplitting, invented context or intensity.
- 44,100 Hz mono, no trimming or amplitude normalization. Prepared train-only statistics.
- Text-only Predictor; character ID enters TTS only. Seven soft emotion probabilities.
- Differentiable speed/pause bounding, tau=0.1/0.02; pitch unchanged.
- All TTS generator layers trained; separate waveform/duration discriminators.
- Batch 1, accumulation 4, epochs 3; initial experimental configuration, not a T4 capacity claim.
- Same-stage strict resume differs from explicit warm-start into joint mode.
- No git repository exists here: no worktree/commit operations. Retain progress in docs.

## Review focus

- Extreme pause logits: finite bounded forward, gradient for the observed negative values.
- Short/long clips: exact source identity, no silent filtering or truncation.
- Multiple optimizer overflow/interruption: no mixed-step durable checkpoint.
- Checkpoint normalizations/mappings differ: refuse reuse before training.
- Posterior reconstruction looks good but inference fails: require reference-free generation.

### Task 1: Style adapter
Files: models/conditioning.py, tests/test_conditioning.py.
Interface: StyleConditioner(normalization, character_count, channels, base_embedding).
`forward(character_ids, emotion_probabilities, continuous_z)` returns g and raw/bounded values.
- [x] Write tests asserting 0<=pause<=1 at extreme values, positive observed speed,
  nonzero gradients at r=-0.0453, identity pitch, and rejection of invalid character IDs.
- [x] Run `.venv/Scripts/python.exe -m pytest tests/test_conditioning.py -q` and observe missing implementation.
- [x] Implement stable soft clipping, normalization buffers, embedding/projection.
- [x] Repeat tests and check unchanged Predictor state dict contract.

### Task 2: Melo adapter and source compatibility
Files: models/conditional_tts.py, training/tts_runtime.py, tests/test_conditional_tts.py.
Interface: ConditionalTTS.forward(batch, probs, z), infer(batch, probs, z).
- [x] Test real small Melo forward/backward with all four style paths and reference-free infer.
- [x] Pin source/model artifacts and verify the expected detach/condition sites.
- [x] Implement explicit g in train/infer, duration g-gradient preservation, strict pretrained load.
- [x] Test bad state dict rejection and finite gradient without any style loss.

### Task 3: Data cache
Files: dataset/tts_data.py, tests/test_tts_data.py.
Interface: cache_audio(row, root, cache); KoreanFrontend(text); TTSData[index]; collate_tts.
- [x] Write real WAV tests for stereo/resampling/hash mismatch and cache invalidation.
- [x] Implement atomic cached audio tensors, pinned Korean frontend, feature cache and collator.
- [x] Compare Korean text features with the upstream Korean inference path and reject inconsistent lengths.
- [x] Run targeted tests; preserve complete-label eligibility and original split.

### Task 4: Losses, engine, inference reports
Files: training/tts.py, training/tts_engine.py, tests/test_tts_training.py.
Interface: TTSObjective.generator/discriminator, train_tts(...), evaluate_tts(...).
- [x] Write tiny actual tensor optimization tests: accumulation tail, uninterrupted versus resumed,
  interruption inside optimizer commit, joint-mode checkpoint identity, deterministic evaluation.
- [x] Implement Melo generator/discriminator losses; AMP unscale/check all gradients before any step.
- [x] Store all optimizers, schedulers, scaler, RNG, stage and cursor in one atomic checkpoint.
- [x] Add full epoch validation, fixed-seed reference-free samples and style comparisons.
- [x] Verify TTS-only gradients reach actual Predictor heads; report true preflight failures.

### Task 5: Colab entry points and delivery
Files: config/tts.json, scripts/colab_train_tts.py, notebooks/08_train_tts.ipynb,
docs/tts-training.md, pyproject.toml, README.md, implementation-progress.md.
- [x] Write self-contained fresh-runtime bootstrap, strict input identity and GPU preflight.
- [x] Default to warmup; explicit joint mode requires warmup and Predictor best weights.
- [x] Expose batch/epochs/output settings and print cache/training progress.
- [x] Export lightweight report + sample audio, keeping model checkpoints on Drive.
- [x] Run full pytest, syntax/notebook parity and real pinned-model loading checks where feasible.
- [x] Build dist/fantasyvoice-pilot.zip and describe exactly which GPU checks remain unexecuted.

## Progress

Tasks 1–5 implemented. Sequential main-session self-review per workspace instructions.
83 tests pass. Real pretrained Melo CPU forward/backward, one GAN update, Korean frontend
parity and reference-free inference succeeded. No Colab/T4 execution or listening approval.
Rulings and exact verification limits are recorded in docs/implementation-progress.md.
