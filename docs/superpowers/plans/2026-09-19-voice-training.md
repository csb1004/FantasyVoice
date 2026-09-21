# FantasyVoice Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task.
> 프로젝트 지침에 따라 메인 작업에서 순차 실행한다. 하위 에이전트는 사용하지 않는다.

**Goal:** Colab에서 Voice 데이터의 샘플 전사부터 스타일 라벨 생성, 준비/공동 학습,
등록된 캐릭터의 reference-free 추론까지 검증 가능한 파이프라인을 만든다.

**Architecture:** 원본 및 자동 분석 결과를 보존하는 dataset 계층과 frozen analyzer를
학습 모델에서 분리한다. KLUE BERT의 연속 스타일 출력을 확장된 MeloTTS에 전달하고,
준비 단계 후 생성기 손실과 스타일 손실로 함께 업데이트한다.

**Tech Stack:** Python, PyTorch, MeloTTS Korean, Whisper large-v3, KLUE BERT,
emotion2vec+ large, torchcrepe, Silero VAD, Google Colab/Drive.

**Spec:** `docs/superpowers/specs/2026-09-19-voice-training-design.md`

## 계획 상태와 실행 경계

2026-09-19 진행: Task 1의 목록 생성과 Task 2의 전사/검수 코드 및 Colab 노트북 구현.
로컬 실제 목록 생성 및 13개 테스트 통과. 실제 Whisper GPU 전사와 사용자 청취 검수는
아직 실행하지 않았으므로 Task 2 전체를 완료로 표시하지 않는다.
현재 테스트 파일은 `tests/test_pilot.py`로 통합했으며 실행 명령은 `python -m pytest -q`이다.
원본 파일 크기와 의존성 분리를 고려해 설정은 승인된 JSON 형식의 `config/pilot.json`을 사용한다.

승인된 모델/학습 방향에 대한 단계별 계획이다. 설계 8절의 미결정 항목을
임의로 채운 실행 명세는 아니다. Task 1의 조사와 Task 2의 ASR 파일럿으로
근거를 마련하고, 각 후속 단계의 선행 결정을 사용자에게 확인한 뒤 상세 구현한다.
모든 단계의 인터페이스를 먼저 코드로 고정하지 않는다.

## Global Constraints

- 첫 버전에서는 현재 대사만 사용한다.
- 감정 강도는 사용자의 명시적인 선택으로 첫 버전에서 보류한다.
- 원본 WAV를 덮어쓰지 않는다.
- 분석기는 라벨 생성 시에만 실행하며 라벨을 캐시한다.
- 캐릭터 ID별 학습 가능한 임베딩을 TTS에만 입력한다.
- 분할과 정규화 통계가 검증/테스트 자료를 학습에 유출하지 않아야 한다.
- 미결정 모델 구조, 손실, 데이터 정책은 코드 작성 전에 확인한다.
- GitHub 작업이 필요하면 PowerShell을 사용하며 gh는 사용하지 않는다.
- 현재 폴더는 Git 저장소가 아니다. 본 문서 작성은 Git 초기화/원격 생성/커밋을 포함하지 않는다.

## Review Focus

1. 비언어 발성에 그럴듯한 문장을 만들어내는 ASR: Task 2에서 청취와 전사를 대조한다.
2. 파일명이 달라도 같은 음성/대사인 표본: Task 3에서 분할 누출을 검사한다.
3. 무음/무성음뿐인 파일: Task 4에서 speed/pitch/pause 결측 및 NaN 방지를 검사한다.
4. TTS 내부에서 끊긴 gradient 또는 무시되는 style: Task 5/6에서 역전파와 조건 변경을 각각 검사한다.
5. Colab 중단 후 다른 캐릭터 ID로 재개: Task 6에서 매핑/설정/통계 불일치를 검사한다.

## Task 1: 원본 조사와 재현 가능한 입력 목록

**예정 파일:** `src/fantasyvoice/dataset/inventory.py`,
`tests/test_inventory.py`, `config/data.yaml`, `docs/data-inventory.md`.

**입출력:** 사용자 지정 Voice root를 읽어 상대 경로, 캐릭터 폴더, sample rate,
channels, frames, duration, 원본 해시, 읽기 실패 사유를 포함한 조사 결과를 생성한다.

- [x] 현재 read-only 헤더 조사 결과를 `docs/data-inventory.md`에 기록했다. 전체 파일 디코딩 검증과 구분한다.
- [x] 기본 wave 리더의 미지원 형식 16개를 호환 디코더로 재확인했다. 모두 FLOAT WAV로 헤더 읽기 성공.
- [ ] 확장자 대소문자, 한글/공백 경로, 손상 WAV, 비음성 파일을 포함하는 작은 fixture로 검사한다.
- [ ] 경로 루트가 바뀌어도 상대 경로/캐릭터 매핑이 동일하도록 정렬한다.
- [ ] 원본을 변경하지 않았는지 fixture 해시를 전후 비교한다.
- [ ] 캐릭터별 분량과 매우 짧은/긴 음성을 보고하여 파일럿 표본 선정 근거를 만든다.

검증 명령: `python -m pytest tests/test_inventory.py -q`.
통과 기준: 읽기 실패를 누락하지 않고 원본 변경 없이 조사 결과를 생성한다.

## Task 2: Colab Whisper 샘플 전사와 청취 검수

**예정 파일:** `src/fantasyvoice/dataset/transcribe.py`,
`notebooks/01_transcription_pilot.ipynb`, `tests/test_transcribe_resume.py`,
`docs/asr-pilot-results.md`.

**선행:** Task 1 목록, 설계 8절의 결과 저장 형식 결정.
**입출력:** 선정한 원본 경로/해시 -> 한국어 ASR 후보, 모델 revision/옵션,
처리 상태/오류, 원본 재생 경로. Whisper 출력은 사람이 확정한 전사와 구분한다.

- [ ] Colab GPU 종류/VRAM, Python/PyTorch/CUDA 및 모델 revision을 기록하는 셀을 만든다.
- [ ] 사용자 Drive 마운트 후 입력 root를 확인하고 출력 root와 구분한다.
- [ ] 캐릭터별로 길이가 다른 표본과 기합/공격 계열 파일을 포함해 파일럿 목록을 제시한다.
- [ ] 한국어 transcription을 실행하고 오디오와 전사 후보를 함께 검토할 수 있게 표시한다.
- [ ] 한 파일 실패 후 재개해 이미 성공한 동일 해시/모델/옵션 항목이 덮어써지지 않는지 검사한다.
- [ ] 모델 로딩 실패/OOM은 실제 실패로 표시한다. 더 작은 모델로 자동 교체하지 않는다.
- [ ] 비언어 발성과 고유명사 오인식을 청취로 확인하고 검수 대기 기준 후보를 정리한다.
- [ ] 결과를 검토받은 뒤 전체 전사 실행 범위와 선별 기준을 결정한다.

검증 명령: `python -m pytest tests/test_transcribe_resume.py -q`.
추가 실제 검증: 사용자의 Colab에서 large-v3 샘플 전사를 실행한다.
로컬 mock 테스트 통과를 실제 GPU 전사 성공으로 보고하지 않는다.

## Task 3: 학습 manifest, 검수 및 데이터 분할

**예정 파일:** `src/fantasyvoice/dataset/manifest.py`, `review.py`, `splits.py`,
`preprocess.py`, `tests/test_manifest.py`, `tests/test_splits.py`.

**선행 결정:** JSONL 스키마, 오디오 전처리, split 비율/seed/중복 그룹 정책,
검수 대기 기준. Task 2 원본 ASR 출력을 입력으로 사용한다.

- [ ] ASR 원문과 수동 수정문, pseudo label과 수동 라벨을 별도 필드로 정의한다.
- [ ] 검수 대기 항목이 training loader에 들어가지 않는 검사를 만든다.
- [ ] 원본 해시 중복과 정규화된 동일 대사 그룹이 split을 넘지 않는지 검사한다.
- [ ] 학습에 자료가 없는 캐릭터와 표본이 너무 적은 캐릭터를 보고한다.
- [ ] 승인된 전처리만 사본에 적용하고 원본과 처리본의 sample rate를 각각 보존한다.
- [ ] 입력/설정/스키마 버전이 바뀌면 기존 캐시를 잘못 재사용하지 않도록 검사한다.

검증 명령: `python -m pytest tests/test_manifest.py tests/test_splits.py -q`.
통과 기준: 원본 추적, 검수 제외, split 재현성 및 누출 방지가 모두 확인된다.

## Task 4: Frozen analyzer와 스타일 target

**예정 파일:** `src/fantasyvoice/analyzers/emotion.py`, `pitch.py`, `speech.py`,
`features.py`, `config/analyzers.yaml`, `tests/test_style_features.py`,
`notebooks/02_style_pilot.ipynb`.

**선행 결정:** 분석기 revision/크기, 감정 분포 변환, F0/VAD 기준,
캐릭터 피치 기준 집계. 입력은 Task 3의 분할된 manifest이다.

- [ ] 감정 원본 9개 점수와 승인된 7개 분포/검수 상태를 보존한다.
- [ ] 피치와 발화 구간을 추출하고 설계 4절 식으로 요약값을 계산한다.
- [ ] 무음, 피치 결측, 발화 한 구간, 여러 구간, 앞뒤 긴 여백 fixture를 검사한다.
- [ ] 검증 표본의 피치를 바꿔도 학습 정규화/캐릭터 기준이 변하지 않는지 검사한다.
- [ ] frozen/eval 상태와 학습 optimizer 미포함을 검사한다.
- [ ] 실제 한국어 표본의 감정/피치/휴지를 청취 및 시각화로 검토한다.

검증 명령: `python -m pytest tests/test_style_features.py -q`.
통과 기준: 결측을 거짓 0 target으로 쓰지 않고 분석 출처 및 통계가 재현된다.

## Task 5: Style Predictor와 MeloTTS 조건 확장

**예정 파일:** `src/fantasyvoice/models/style_predictor.py`, `melo_adapter.py`,
`training/losses.py`, `config/model.yaml`, `tests/test_losses.py`,
`tests/test_joint_gradient.py`.

**선행 결정:** 조건 주입 위치, 계층 학습 범위, 체크포인트 호환 전략.
MeloTTS 원본 코드를 실제 고정 revision으로 읽고 그 구조에 맞춰 상세 설계를 확정한다.

- [ ] BERT 입력을 text-only로 제한하고 감정 7개 확률 및 연속값 3개를 출력한다.
- [ ] 감정 확률 가중합과 연속 스타일을 캐릭터 임베딩과 구분해 조건으로 전달한다.
- [ ] 원래의 한국어 가중치 로딩에서 허용한 신규/크기 변경 항목만 명시적으로 처리한다.
- [ ] Soft CE, 표준화 Smooth L1, 유효 표본 평균, 전부 결측인 배치를 검사한다.
- [ ] 실제 TTS loss만으로 Predictor에 gradient가 전달되는지 검사한다.
- [ ] analyzer와 판별기 업데이트 경계의 의도적인 detach는 유지한다.
- [ ] 참조 음성 없이 추론하는 경로와 조건을 무시하지 않는지 확인한다.

검증 명령: `python -m pytest tests/test_losses.py tests/test_joint_gradient.py -q`.
실제 모델 GPU 검사와 가벼운 단위 검사를 명확히 분리한다.

## Task 6: 준비/공동 trainer, 재개 및 평가

**예정 파일:** `src/fantasyvoice/training/trainer.py`, `checkpoint.py`,
`validation.py`, `config/training.yaml`, `tests/test_training_resume.py`,
`notebooks/03_joint_training.ipynb`.

**선행 결정:** 학습률/동결 범위/준비 단계 길이, batch 및 학습 종료 기준.

- [ ] 준비 단계에서는 pseudo style, 공동 단계에서는 predicted style을 입력하는 검사를 만든다.
- [ ] Generator/Predictor와 discriminator 업데이트를 분리하고 loss/gradient를 기록한다.
- [ ] 선택한 작은 실제 데이터로 한 번의 업데이트와 유한한 loss를 확인한다.
- [ ] checkpoint에 모델, optimizer, scheduler, scaler, RNG, 단계, step,
  캐릭터 매핑, 정규화 통계 및 설정/revision을 저장한다.
- [ ] 재시작 후 단계/캐릭터/통계를 복원하고 불일치 설정에서는 명확히 중단한다.
- [ ] 동일 문장/캐릭터에서 스타일 조건만 바꾸어 reference-free 출력을 비교한다.
- [ ] pseudo label 일치도와 사람 청취 결과를 구분해 보고한다.

검증 명령: `python -m pytest tests/test_training_resume.py -q`.
실제 검증: Colab 작은 학습 -> 저장 -> 런타임 재시작 -> 재개 -> 추론.

## Task 7: 추론과 실행 가이드

**예정 파일:** `src/fantasyvoice/inference/synthesize.py`,
`notebooks/04_inference.ipynb`, `tests/test_inference_contract.py`, `README.md`.

- [ ] text와 등록된 character ID만으로 Predictor/TTS를 호출하는 진입점을 만든다.
- [ ] 미등록 캐릭터, 빈 대사, 미지원 intensity는 명확한 오류로 처리한다.
- [ ] 추론 시 analyzer/ASR/reference audio를 요구하지 않는지 검사한다.
- [ ] Drive 경로, 검수, 순차 실행, 체크포인트 재개 및 생성 파일 위치를 문서화한다.
- [ ] 완료 보고에서 미실행 Colab 검사, 실제 완료한 파일럿, 전체 학습 여부를 구분한다.

검증 명령: `python -m pytest tests/test_inference_contract.py -q`.

## 자체 검토

- 승인된 모델, 강도 제외, context 제외, character 입력 경계와 warmup/joint 전환을 반영했다.
- 분석기와 생성 모델 학습을 분리하고 pseudo target 출처를 보존한다.
- GPU 실행 전과 후의 검증을 구분했다.
- 실제 Melo adapter 설계 없이 gradient가 보장된다고 주장하지 않았다.
- 미결정 항목은 선행 조건으로 명시했으며 가짜 모델 구현으로 대체하지 않는다.
