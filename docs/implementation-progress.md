# Execution ledger — 2026-09-19-voice-training

- User approved JSONL, design direction and implementation of inventory + ASR pilot.
- Applying executing-plans, test-driven-development and verification-before-completion.
- No Git repository exists; worktrees/commits are inapplicable. Work in the current workspace.
- User requires sequential work; no delegated agents.
- Tasks 3–7 retain their documented prerequisite decisions; no speculative trainer is created.
- Inventory and pilot share relative paths + SHA256; review uses transcription result key.
- Pilot sampling is an audit aid, not an automatic training selection policy.
- ASR thresholds are Whisper defaults, not dataset acceptance thresholds; all pilot outputs await review.
- Inventory + ASR pilot implementation written; first tests failed on missing package as expected.
- Path relocation test exposed absolute path in libsndfile errors; store stable decoder error text.
- Recovery test exposed stale successful history overriding a later failure; resume now uses latest status.
- Local full inventory: 25,875 readable WAV headers, 89 characters, 18.8048918 hours, 0 header errors.
- Previously unreadable 16 files are stereo FLOAT; 293 pilot candidates selected.
- Notebook code cells parsed, CLI inventory executed against actual Voice data.
- Actual Whisper/Colab execution and listening review remain unexecuted; no model quality claim.
- Verification: `python -m pytest -q`: 13 passed, including float/stereo/corrupt sources,
  relocation, byte preservation, resume/config mismatch, source change/recovery, interruption,
  output boundaries, OOM, review history, and failed atomic replacement.
- Code compiled; code-only bundle extracted into a temporary directory and its suite passed (13 tests).
- Bundle contains no Voice, Announce, SD Character, output data or WAV files.
- Pilot code is ready for Colab; Tasks 3–7 are not implemented and need the recorded decisions.

## User-supplied Colab result audit

- Inspected `pilot-v1.zip` without modifying its contents.
- Runtime evidence: Tesla T4, 14.56 GiB, Python 3.13.15, Torch 2.11.0+cu128.
- Whisper large-v3 produced 12 successful transcript records; no errors recorded.
- This supersedes the earlier unexecuted-Colab status for those 12 samples only.
- Remote inventory contains 796 files / 18 characters, not the complete local dataset.
- All 796 remote file hashes match local sources; one sampled filename differs by NFC/NFD encoding.
- No reviews.jsonl supplied, all transcript review states pending, no listening performed here.
- Four ASR segment endpoints exceed audio length; they are not reliable alignment targets.
- Findings: `docs/asr-pilot-results.md`; structured audit: `outputs/pilot-audit/audit.json`.
- Await clarification of intended remote data scope and listening review results before downstream decisions.

## 2026-09-20 Drive inventory performance fix

- User reported 1,500 files scanned in 12 minutes. Code scan hashes every entire WAV over Drive
  before choosing the 12-file pilot; at that rate a full pass projects to about 207 minutes.
- Default notebook now reads the bundled local reference inventory; explicit scan mode remains available.
- Reference mode does not claim the full remote dataset exists. Origin and summary record that distinction.
- Selected sources still undergo byte hash verification in run_pilot; no model/label/training changes.
- Added NFC/NFD path lookup for the filename difference already observed in the user's results.
- Added tests before implementation; 15 tests pass. Actual Colab performance is not yet measured.

## Silent playback correction

- Reported Adela_PlaySkill1024200seq0_4_ko.wav: 2,772 frames at 44,100 Hz,
  approximately 0.062857 seconds, all samples exactly zero, valid PCM16 file.
- Default IPython array normalization is unsuitable for all-zero signals.
- Preview now detects empty/silent/nonfinite audio, preserves quiet signal amplitude,
  and encodes browser PCM16 bytes. Source audio and transcription are unchanged.
- UI resets the selected text before loading audio, catches playback errors, and displays causes.
- Added a review-only Colab refresh script so ASR does not need to run again.
- No ASR model, training eligibility threshold or automated rejection policy changed.

## Imported listening reviews and user-approved duration cutoff

- User supplied D:/DATA/Downloads/reviews.jsonl: 17 events, 12 unique keys.
- Latest decisions: accepted 5, rejected 1, pending 6. All keys reconstructed using
  existing run configuration, inventory hashes and relative paths.
- 10/12 reviewed clips are <=0.5 seconds, including all five accepted clips.
- Preserved review history and decisions; no pending/rejected item automatically accepted.
- Implemented the agreed <=0.5 second exclusion before every pilot selection branch.
- Full local inventory: 1,121 excluded by duration, 24,754 pass duration only.
- 19 tests passed. Future training manifest must apply this policy; training remains unimplemented.
- Report: docs/listening-review-results.md. Linked metadata: outputs/review-audit/.
- Adriana corrected text is still rejected; user confirmation needed to interpret intent.

## Full dataset stage

- User confirmed Adriana audio is suitable; appended accepted amendment, retained original history.
- User requested full dataset preparation rather than continued 12-clip pilots.
- Added full_dataset runner: all 24,754 duration-eligible local references, no sample limit.
- Stage one file locally, verify its SHA256, screen exact silence/invalid audio, then run approved ASR.
- 100-source result shards checkpoint every 20 attempts. Normal interruption flushes results;
  hard termination can repeat at most the uncheckpointed 19-file tail.
- Resume skips completed results for the immutable reference snapshot; errors are retried.
- Exports full candidates, human-confirmed subset, issues, short exclusions and progress.
- Existing review corrections merged by source identity and timestamp. No pending review auto-approved.
- Added full Colab one-cell script and notebook. No runtime execution against actual full corpus yet.
- 26 tests passed, including bounded checkpoints, interruption/retry, pilot reuse,
  silence handling, review amendments, immutable configuration/reference, existing pilot/playback tests.
- Full automatic quality acceptance thresholds, style labels and train/validation split remain undecided.

## Post-transcription code prepared while user's ASR continues

- User reports full ASR is progressing and requested code to run after completion.
- Added finalize_dataset: validates original run/inventory identity and checkpoint records,
  distinguishes missing/error/review statuses, reapplies reviews and exports an audio-free report ZIP.
- Added CPU-only scripts/colab_after_dataset.py and notebooks/03_after_dataset.ipynb.
- Script snapshots metadata locally, detects progress changes during copying, leaves original
  dataset-v1 outputs untouched, and saves a complete or explicitly partial report archive to Drive.
- No ASR models loaded and no source WAVs read by postprocessing.
- No new analyzer, loss, split or quality threshold was selected.
- 30 local tests passed including finalization, stale exports, partial runs, review updates,
  snapshot mismatch refusal and archive audio exclusion. Actual Colab postprocessing not run yet.

## Provisional style labeling implementation (2026-09-20)

- User explicitly requested implementation against the current format while ASR continues,
  allowing later adjustment. No running ASR files or source WAVs were modified.
- Added frozen emotion2vec_plus_large, torchcrepe and Silero adapters plus the pinned Melo
  Korean frontend. Configurable extraction defaults are explicitly provisional.
- Nine emotion probabilities retained; unsupported top emotions mask the seven-class target.
  Speed uses phoneme count / VAD speech seconds; pause excludes edge silence; pitch stores Hz.
  Character-relative pitch, standardization, intensity, data splits and training remain deferred.
- Resumable hashed checkpoints, source SHA verification, review preservation, changed-text
  invalidation, retryable failures, OOM stop and package/code provenance implemented.
- Added Colab script/notebook 04 reading finalized report candidates, with one-file model
  preflight and separate style-v1 output. Full candidates processed, no pilot-size restriction.
- 40 local tests passed including real WAV decoding/resampling and torch tensor masking with
  substituted analyzer outputs. Actual downloaded analyzer models / Colab GPU not executed.

## Colab Silero import dependency fix (2026-09-20)

- User's Python 3.13 Colab failed during FrozenAnalyzers initialization: silero_vad's
  sequence_vad imports onnxruntime unconditionally although package metadata marks it optional.
- Added explicit CPU onnxruntime==1.24.4 dependency; PyPI provides Linux x86_64 CPython 3.13 wheels.
- Existing sessions can install the missing package and rerun 04 without changing data or models.

## Colab Python 3.13 MeCab initialization fix (2026-09-20)

- User reported g2pkk annotate calling None.pos. Upstream get_mecab swallows initialization
  errors; python-mecab-ko 1.3.7 has no CPython 3.13 Linux wheel.
- Added Python-version-specific Linux MeCab dependency and pinned Korean dictionary.
  Compared original and py313 compatibility wheel wrappers: only version string differs.
- Added explicit analyzer preflight and reset of a cached failed Melo G2P object on retry.
- Regression tests cover failure reporting and invalid cached object recovery. Linux binary
  execution / full Colab analyzer pipeline still requires runtime verification.

## Silero TorchScript freezing fix (2026-09-20)

- User reached model loading and exposed unsupported ScriptModule.requires_grad_.
- Replaced module-wide freezing with eval plus requires_grad_(False) per parameter.
- 44 tests passed, including actual torch.jit.script module freezing and inference.
- Downloaded the hash-verified official Silero 6.2.2 wheel, reproduced the original exception
  with its silero_vad.jit, then verified corrected freezing and CPU inference on a 512-sample input.
- This verifies Silero CPU initialization/inference, not the entire GPU labeling pipeline.

## Post-style validation/export (2026-09-21)

- User reported the 04 cell finished and requested the next code.
- Added CPU-only checkpoint-based finalization with candidate/config/text key verification,
  missing/error/partial distinctions, numeric/target-mask validation, distributions and per-character counts.
- Source reviews and splits remain unchanged; no acceptance threshold, split, normalization or trainer added.
- Added script/notebook 05: copy metadata to a temporary snapshot, detect changes during copy,
  export a report ZIP to Drive and download it. No models, source WAVs or GPU are needed.
- Regression coverage: stale exports, failed/missing records, partial emotion, invalid values,
  edited-text cache invalidation, archive contents and source preservation.

## Prepared 06 implementation pending policy answers (2026-09-21)

- User requested code after lightweight 05 as well. Asked split ratio, candidate eligibility,
  pitch baseline/seed questions; no answers applied yet. Settings deliberately remain null.
- Added explicitly configured group-aware manifest splitting, train-only character F0 median
  baselines, continuous-target standardization and missing-target masks. No trainer/audio conversion.
- Added 06 Colab entry point and documentation. It refuses to run without confirmed settings
  and preserves prior output ZIPs. No actual user dataset has been split or approved.

## Approved policies and actual preparation (2026-09-21)

- User selected proposal 1 after reviewing style-report.zip: complete labels only,
  speed >30 phonemes/s held out, 95:5 split, seed 42, train-only character median F0.
- Applied the speed filter before splitting/statistics; added strict-boundary and invalid-limit tests.
- Independently checked source style keys and numeric invariants against the supplied report.
- Created outputs/dataset-prepared-v1.zip: 19,784 train, 1,040 validation, 2,726 held
  (2,590 incomplete targets and 136 speed outliers). All 89 characters appear in both splits.
- Human review states and raw pseudo labels are preserved. Models have not been trained;
  source audio preprocessing and trainer architecture/configuration remain future work.

## 07 Predictor warmup implementation (2026-09-21)

- User approved full KLUE BERT + heads, explicitly treating batch/epochs as adjustable initial experiments.
- Added pinned KLUE BERT CLS predictor (unused pooler omitted), masked soft-CE/SmoothL1,
  full-encoder AdamW training, FP16/gradient accumulation, train-only constant evaluation baseline.
- Added deterministic update-boundary resume with optimizer/scheduler/scaler/RNG and input/config identity.
  A checkpoint interrupted inside optimizer.step cannot replace the last durable valid state.
- Added T4 Colab 07 with configurable knobs, small real gradient preflight, latest/best weights,
  validation predictions, original-unit errors and a lightweight report archive.
- Local real small Transformers BERT test passes; full downloaded KLUE BERT T4 training remains unverified.
- Ruling: settings changes require a new experiment directory and pretrained restart, preventing
  silent changes to schedule/optimizer semantics. Future warm-start extensions are separate work.
- Verified all 19,784/1,040 actual prepared rows with the pinned KLUE tokenizer (max tokens 71/60).
- Downloaded full pretrained pinned KLUE BERT (110,034,442 predictor parameters); actual-data
  CPU forward/backward preflight passed on two samples. This is not full training or a T4 memory test.
- Added tests for FP16-overflow control-flow replay (simulated scaler) and validation-interruption retry.
- Final verification: 67 tests passed; Python syntax and 07 notebook/script parity checked.

## Actual 07 report audit (2026-09-21)

- User supplied predictor-v1-report.zip: T4 training completed 3 epochs / 1,857 updates.
- Best full validation at update 1,238 (epoch 2): 2.4965426 versus constant baseline 3.034362.
- 250/1,040 raw pause predictions were negative. Raw v1 weights/output semantics remain unchanged.
- Report contains no weights; the user's best-model.pt itself was not available locally.

## 08 Conditional TTS implementation (2026-09-21)

- User approved conditional-tts-design option 1. Added smooth speed/pause bounds at the TTS
  boundary, character/soft-emotion/continuous embeddings and explicit g routing through Melo.
- Preserved original Predictor model and its supervision. Duration predictors retain x detach
  but allow g gradients. Original speaker lookup remains frozen/unused for base load compatibility.
- Pinned Korean Melo source and model revision; verified source and pretrained artifact hashes,
  strict model state loading (51,899,697 parameters including conditioner).
- Audio cache verifies and decodes the same bytes from one Drive read, resamples to 44.1kHz mono,
  preserves amplitude and silence; text cache keys include text, source and frontend versions.
- Frozen Korean BERT feature output matched upstream exactly on a real Korean sentence (max diff 0).
- Replaced Melo's per-item Fourier-matrix verification with equivalent complex STFT;
  independent direct-convolution comparison passes at atol 1e-4.
- TTS warmup and explicit joint mode implemented with waveform/duration discriminators,
  accumulation, all-optimizer finite-gradient check, overflow replay, strict same-stage resume,
  pending evaluation recovery, full epoch validation and reference-free sample generation.
- Actual pretrained Melo + local real WAV + actual Korean frontend: CPU TTS-only backward
  reached all four style outputs and KLUE Predictor heads. This used pretrained KLUE with fresh
  heads, not the user's unavailable trained v1 checkpoint. One real CPU GAN update of all three
  optimizers completed; text+character inference produced 118,272 finite audio samples.
- Final self-review performed sequentially per workspace instructions; no independent agent review.
  Extreme negative-speed underflow was reproduced and fixed with the minimum positive FP32 floor.
  Pause derivative at 0/1 is tested; observed negative-pause gradients remain nonzero.
- Ruling: MAS/flow/duration/spectral computations remain FP32; decoder, discriminators and joint
  Predictor use CUDA autocast. This trades some memory for distribution stability.
- Ruling: GAN optimizers use gradients from the same accumulated window before any step;
  CPU tests verify transactional resume and overflow control flow. CUDA numeric behavior is unverified.
- Ruling: duration discriminator uses target duration as real and prediction as fake, correcting the
  reversed argument ordering in the pinned upstream training script. Its unused g projection is omitted.
- Initial settings remain experimental. Full Colab/T4 execution, peak GPU memory, full corpus training,
  learned-v1 integration and listening quality are not verified locally.
- Local suite: 83 passed, 6 upstream deprecation warnings. Syntax/notebook parity and bundle checked.
