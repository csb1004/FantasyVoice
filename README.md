# FantasyVoice

한국어 대사에서 스타일을 예측하고 캐릭터 음성을 생성하기 위한 학습 파이프라인입니다.
**음원 목록 → Whisper 전사 → 검수·스타일 라벨 → 데이터 분할 → Style Predictor → 조건부 MeloTTS 준비·공동 학습** 코드가 구현되어 있습니다.
노트북은 `01`부터 `12`까지 단계별로 제공합니다. CPU 모델 검증과 실제 GPU 학습·청취 품질 검증은 구분해 기록합니다.

이 저장소에는 코드·노트북·설정·문서·테스트만 포함합니다. 원본 음원·이미지,
학습 데이터와 보고서, 모델 가중치, 로컬 캐시·배포 ZIP은 포함하지 않습니다.
기존 데이터 준비와 07 학습을 마쳤다면 [08 실행 안내](docs/tts-training.md)부터 진행하세요.
08 준비 학습까지 완료했다면 [09 공동 학습 안내](docs/joint-training.md)를 사용하세요.
09 공동 학습 완료 후에는 [10 음성 생성 안내](docs/inference.md)를 사용하세요.
감정 분석기를 먼저 지도학습한 뒤 고정하여 TTS에 감정 loss를 추가하려면 [11·12 실행 안내](docs/emotion-feedback.md)를 사용하세요.

## 전체 데이터셋 준비 (12개 제한 없음)

최신 `dist/fantasyvoice-pilot.zip`을 Drive의 `MyDrive/FantasyVoice/`에 교체하고,
GPU 런타임을 재시작한 뒤 `notebooks/02_build_dataset.ipynb`의 한 셀을 실행하세요.
기존 파일럿 노트북의 12개 설정을 바꾸는 방식이 아니라 전체 전사용 실행 파일을 사용합니다.
파일럿 노트북의 `PILOT_LIMIT=0`은 24,754개 전체를 뜻하지 않습니다.

현재 로컬 목록의 25,875개 중 **0.5초 초과 24,754개 전부**를 대상으로 처리합니다.
추가 파일/변경 파일이 있다면 로컬 목록과 ZIP을 갱신해야 합니다.
원본은 그대로 두고 처리할 파일 하나씩 Colab 임시 저장소로 복사해 해시 확인 후 전사합니다.
실제로 업로드되지 않은 파일, 원본 해시가 다른 파일은 오류로 기록합니다.
완전 무음/빈 파형/비정상 수치 및 빈 전사는 검수 대기로 보관합니다.
0.5초를 넘는 기합이나 잡음까지 자동 판별하는 것은 아닙니다.

출력: `MyDrive/FantasyVoice/dataset-v1/`

| 파일 | 내용 |
|---|---|
| `candidates.jsonl` | 전사 성공 후보 전체. 수정문과 ASR 원문을 함께 보존. 미검수 상태 명시 |
| `reviewed.jsonl` | 사람이 대사를 확인한 후보만. 최종 음향/스타일 품질 합격을 뜻하지 않음 |
| `needs-review.jsonl` | 무음, 빈 전사, 처리 오류, 사람이 부적합 판정한 항목 |
| `excluded-duration.jsonl` | 0.5초 이하 제외 목록 |
| `results/00000.jsonl` 등 | 100개 원본 단위 묶음의 전사 및 재시도 이력 |
| `progress.json` | 전체/성공/검수대기/오류/미처리 개수 |
| `export-summary.json` | 후보와 사람 확인 수, 미완료 후속 단계 |

20개 처리마다 저장합니다. 일반적인 실행 중지/예외에서는 완료된 결과를 저장하고 내보냅니다.
런타임 강제 종료/연결 장애 시에는 마지막 저장 이후 최대 19개를 다시 처리할 수 있습니다.
같은 셀을 다시 실행하면 저장된 성공/검수대기 항목은 건너뛰고 오류/미처리를 실행합니다.
완료 항목은 재개 시 Drive에서 다시 읽지 않습니다. 원본이 바뀌었다면 기준 목록을 갱신하고
새 출력 폴더를 사용하세요. 모델 설정/기준 목록이 바뀐 채 같은 출력 폴더에 재개하면 중단합니다.
한 출력 폴더는 동시에 하나의 런타임에서만 사용해야 합니다.

이전 파일럿 전사는 설정과 원본 해시가 일치하면 재사용합니다. Drive의 파일럿 검수 결과와
사용자 확인을 반영한 ZIP의 검수 기록은 시간순으로 병합하며 기존 파일을 덮어쓰지 않습니다.
원본 오디오가 같다면 모델 설정이 달라져도 사람의 대사 수정은 유지할 수 있습니다.
새로 검수한 내용을 반영하려면 같은 실행을 다시 하면 완료된 ASR을 반복하지 않고 다시 내보냅니다.

**자동 전사 후보를 모두 정답으로 승인하지는 않습니다.** 자동 전사 품질 기준,
감정/운율 라벨 생성 및 학습 분할을 정한 뒤 실제 학습 입력으로 확정해야 합니다.

## 파일럿만 실행하기

전체 전사를 진행 중이라면 아래 파일럿 단계를 다시 실행할 필요는 없습니다.

1. `dist/fantasyvoice-pilot.zip`을 Drive의 `MyDrive/FantasyVoice/`에 올립니다.
2. `notebooks/01_transcription_pilot.ipynb`를 Google Colab에서 엽니다.
3. GPU 런타임을 선택하고 위에서부터 셀을 실행합니다. Drive 연결은 사용자가 승인합니다.
4. 기본 음성 경로는 `/content/drive/MyDrive/Voice`입니다. 폴더가 다르면 설정 셀을 수정합니다.
5. 샘플 목록을 먼저 확인하고 전사 셀을 실행합니다. 전사에는 모델 다운로드와 GPU 시간이 듭니다.
6. 검수 셀에서 파일을 선택해 음성을 듣고 대사를 수정한 뒤 판정을 저장합니다.

### Drive 목록 생성이 느린 경우

2026-09-20 업데이트: 기본 `INVENTORY_MODE = 'reference'`는 ZIP에 포함된 로컬
목록을 읽습니다. 25,875개 음성을 Drive에서 모두 다운로드해 해시를 계산하지 않습니다.
실제 전사 대상으로 선택한 파일만 전사 직전에 원본 SHA256과 비교합니다.
업로드 누락이나 내용 변경은 오류로 기록하며 자동으로 다른 음성으로 교체하지 않습니다.
한글 파일명의 NFC/NFD 차이는 경로 탐색 시 처리하고 최종 파일은 해시로 검증합니다.

이 모드의 전체 개수는 **로컬 목록 기준**이며 Drive 업로드 완료 검증이 아닙니다.
`inventory-origin.json`과 `summary.json`에 이 차이를 기록합니다.
원본 데이터가 달라졌으면 로컬 목록/ZIP을 갱신하거나, 의도적으로 `scan` 모드를 선택하세요.
업데이트하려면 실행 중인 목록 셀을 중지하고 새 ZIP을 Drive에 교체한 뒤,
새 노트북을 열고 런타임을 재시작해 설치 셀부터 실행합니다. 기존 결과 폴더는 보존합니다.

노트북은 **0.5초 이하 파일을 제외한 뒤** 캐릭터별 길이 분포에서 3개와, 존재하면 공격/피격 계열 1개를
추가 선택합니다. 일부 선택이 겹칠 수 있습니다. 처음에는 그중 12개를 서로 다른 캐릭터에서
골라 실행합니다. 이는 파일럿 표본이며 학습 데이터 자동 선별 기준이 아닙니다.
길이 기준 제외 파일은 `excluded-duration.jsonl`에 기록하고 원본은 보존합니다.
수정문이 실제 음성과 일치한다면 '대사 확인'으로 저장합니다. 자동 전사가 처음에 틀렸다는
이유만으로 '부적합'을 선택할 필요는 없습니다. 보류 상태로 저장하면 검수 이력은 남지만
대사를 확인한 것으로 집계되지 않습니다.
원하면 노트북의 `PILOT_LIMIT`를 늘려 전체 파일럿 목록을 확인합니다.

## 로컬 목록 생성

Python 3.10 이상을 사용합니다. Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[dev]'
.\.venv\Scripts\python.exe -m fantasyvoice inventory --root Voice --output outputs/inventory
.\.venv\Scripts\python.exe -m pytest -q
```

로컬 CLI의 목록 생성은 모델 다운로드나 GPU를 요구하지 않습니다. WAV 헤더와 SHA256을 읽습니다.
libsndfile을 사용하므로 PCM과 부동소수점 WAV를 읽을 수 있습니다.
전체 파형 디코딩, 녹음 품질, 음성 존재 여부까지 검증했다는 뜻은 아닙니다.

## 결과 파일

| 파일 | 의미 |
|---|---|
| `inventory.jsonl` | 상대 경로, 캐릭터 폴더, SHA256, 오디오 형식/길이, 오류 |
| `summary.json` | 캐릭터별 분량, 형식 및 오류 통계 |
| `pilot.jsonl` | 재현 가능한 파일럿 표본 목록 |
| `run.json` | ASR 가중치 해시, 구현 버전, 전사 옵션 |
| `transcripts.jsonl` | 자동 전사 원문/구간/지표, 성공·실패 및 시도 이력 |
| `reviews.jsonl` | 사람이 수정한 대사와 판정의 이력. 원래 ASR 출력을 덮어쓰지 않음 |

검수 판정은 accepted(대사 확인), rejected(부적합), pending(보류)입니다.
대사를 accepted로 판정해도 향후 감정/음향 검사까지 통과한 학습 데이터라는 뜻은 아닙니다.
전사 수치만으로 자동 accepted 처리하지 않습니다.

## 재개와 실패 처리

같은 파일 해시·모델·옵션으로 다시 실행하면 성공한 항목은 건너뛰고 실패한 항목을 재시도합니다.
파일 내용이 바뀌면 목록을 다시 생성해야 합니다. 모델/옵션/구현 버전이 바뀌면 새 결과
디렉터리를 사용합니다. 검수는 원본 해시와 실행 설정이 포함된 키에 연결됩니다.
OOM은 기록 후 중단하고 다른 모델로 자동 교체하지 않습니다.
Whisper 모델 로딩 실패는 전사 시작 전에 오류로 표시됩니다.

한 결과 디렉터리는 동시에 하나의 런타임에서만 사용하세요. 파일마다 결과 snapshot을
임시 파일로 쓴 뒤 교체합니다. Google Drive 마운트의 원격 동기화 완료까지 보장하는 것은
아니므로 종료 전에 Drive에서 저장 결과를 확인하세요.

Whisper 입력 디코딩의 16kHz/mono 변환은 ASR 내부 처리이며 원본을 수정하지 않습니다.
TTS 학습용 전처리 정책을 확정한 것은 아닙니다. `config/pilot.json`의 threshold는
Whisper의 디코딩 기본값을 명시한 것이며 데이터 합격/탈락 임계값이 아닙니다.
`condition_on_previous_text`는 한 파일 안의 Whisper 처리 창 사이에만 적용됩니다.
서로 다른 파일을 문맥으로 이어 붙이지 않습니다.

## 다음 단계

### 전체 전사가 끝난 뒤

`notebooks/03_after_dataset.ipynb`의 한 셀 또는 `scripts/colab_after_dataset.py`를 실행합니다.
현재 실행 중인 전사가 종료된 뒤에만 실행하세요. CPU 런타임에서도 동작하며
Whisper를 불러오거나 원본 음성을 재전사하지 않습니다.

- 저장된 결과와 기준 목록의 해시/실행 설정을 대조합니다.
- 미처리와 오류는 `retry.jsonl`로, ASR 구간이 오디오 길이를 넘는 항목은
  `segment-overruns.jsonl`로 기록합니다. 타임스탬프 이상만으로 음성을 자동 제외하지 않습니다.
- 최신 파일럿/전체 검수 기록을 반영해 후보와 사람 확인 데이터를 다시 내보냅니다.
- 원래 `dataset-v1` 파일은 유지하고 새 결과 ZIP만 Drive의 FantasyVoice 폴더에 저장합니다.
- 전체 ASR 처리가 끝났으면 `dataset-report.zip`, 오류/미처리가 남았으면
  `dataset-report-partial.zip`입니다. 무음 등 검수 대기는 처리 실패와 구분합니다.
- ZIP에 원본 WAV는 포함하지 않습니다. 집계/후보/검수/전사 구간/오류 기록을 포함합니다.
- 자동 브라우저 다운로드가 차단되면 Drive에서 ZIP을 직접 내려받으세요.

결과 ZIP을 다음 작업에 전달하면 전체 분포와 오류를 확인할 수 있습니다.
이 검사는 전사 정확도나 감정/운율 라벨 품질을 보증하지 않습니다.

[설계](docs/superpowers/specs/2026-09-19-voice-training-design.md)와
[계획](docs/superpowers/plans/2026-09-19-voice-training.md)에 합의한 전체 파이프라인 및
후속 구현 전에 결정할 항목을 기록했습니다. 샘플 청취 결과를 바탕으로 전사 검수 기준,
음향 분석 설정, 학습 분할 및 조건 주입 위치를 정합니다.

검증 현황은 [실행 기록](docs/implementation-progress.md)을 참조하세요.

## 캐릭터·스타일 조건 TTS 학습 (08)

최신 코드 ZIP을 Drive에 업로드하고 `notebooks/08_train_tts.ipynb`를 새 GPU
Colab에서 실행합니다. 기본 모드는 실제 MeloTTS 준비 학습이며, 완료 후 별도로
`STAGE = 'joint'`를 선택하면 Predictor와 공동 학습합니다.
07의 `best-model.pt`와 기존 준비 데이터셋을 재사용합니다.
실행 순서, 결과 파일, 재개 및 초기 설정은 [TTS 학습 안내](docs/tts-training.md)를 참고하세요.

## 스타일 라벨 생성 코드

현재 후보 JSONL에 감정·속도·피치·휴지 비율을 추가하는 코드가 준비되어 있습니다.
전체 전사와 위 후속 점검이 끝난 뒤, 최신 코드 ZIP을 Drive에 올리고
새 GPU Colab에서 `notebooks/04_style_labels.ipynb` 또는
`scripts/colab_style_labels.py`를 실행하세요. 입력은 `dataset-report.zip`,
출력은 `MyDrive/FantasyVoice/style-v1/style-labels.jsonl`입니다.

20개마다 저장하며 재실행 시 이어갑니다. 추출 설정은 `config/style.json`에서
수정할 수 있습니다. 검수 상태는 유지하고, 상대 피치 정규화와 학습용 데이터 분할은
후속 단계로 남깁니다. [필드 설명과 재개 방법](docs/style-labels.md)을 참고하세요.
실제 분석기 가중치로 Colab GPU 실행은 아직 검증 전입니다.

## 스타일 라벨 완료 후 점검 (05)

04가 종료되면 최신 코드 ZIP을 Drive에 업로드하고 `notebooks/05_after_style.ipynb`
또는 `scripts/colab_after_style.py`를 실행하세요. CPU에서도 동작하며 모델 설치나
원본 음원 접근 없이 저장된 체크포인트와 입력 후보를 대조합니다.
감정·운율 분포, 누락/오류, 목표값 가용 여부를 점검하고
`MyDrive/FantasyVoice/style-report.zip`을 저장·다운로드합니다.
오류/미처리/형식 이상이 있으면 `style-report-partial.zip`입니다.
partial 라벨은 자동 제외하지 않으며, 데이터 분할·정규화·학습은 수행하지 않습니다.
04에서 후보 경로를 변경했다면 05의 `CANDIDATES_OVERRIDE`도 같은 경로로 설정하세요.

## 후속 데이터 목록 준비 (06)

`notebooks/06_prepare_training_data.ipynb`와 `scripts/colab_prepare_training_data.py`도
준비되어 있습니다. 승인된 설정(95:5, seed 42, 완전한 스타일 라벨,
속도 30음소/초 초과 보류, 캐릭터별 학습 F0 중앙값)이
`config/training-data.json`에 반영되어 있습니다.
CPU에서 학습/검증 목록과 train 기준 정규화 통계를 만들어
`MyDrive/FantasyVoice/dataset-prepared-v1.zip`으로 저장합니다.
[설정과 처리 방식](docs/training-data-preparation.md)을 참고하세요.

## 실제 스타일 예측기 학습 (07)

새 T4 Colab에서 `notebooks/07_train_predictor.ipynb` 또는
`scripts/colab_train_predictor.py`를 실행합니다. 준비된 데이터 ZIP과 최신 코드 ZIP이
필요합니다. KLUE BERT 전체와 감정·연속값 출력층을 함께 학습하며 배치/epoch/LR은
노트북 상단에서 조정할 수 있습니다. 원본 음성이나 04 분석기는 사용하지 않습니다.

동일 설정으로 중단 후 재개할 수 있고, 설정을 바꾼 실험은 새 출력 폴더를 사용합니다.
완료 후 `predictor-v1-report.zip`으로 검증 결과를 확인합니다. 아직 음성을 생성하는
TTS 학습 단계는 아닙니다. [실행·재개·결과 설명](docs/predictor-training.md)을 참고하세요.
