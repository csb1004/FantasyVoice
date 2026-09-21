# 06 데이터 목록 준비

05 style-report.zip의 검증된 라벨로 학습/검증 JSONL, 캐릭터 ID 매핑,
학습 데이터에만 맞춘 피치 기준과 표준화 통계를 만듭니다. CPU에서 실행됩니다.
원본 WAV 리샘플링이나 실제 TTS/Predictor 학습은 아직 수행하지 않습니다.

## 먼저 확인할 설정

2026-09-21 사용자가 제안 1번을 승인하여 `config/training-data.json`을 확정했습니다.
설정이 누락된 경우에는 출력을 생성하지 않습니다. Colab의
`SETTINGS_OVERRIDE`에 같은 구조의 dict를 지정할 수도 있습니다.

| 설정 | 의미 / 지원 값 |
|---|---|
| validation_fraction | 0.05 (학습 95% / 검증 5% 목표) |
| seed | 42 |
| allow_pending_transcripts | true: 자동 전사 후보 사용, 사람 검수 상태는 보존 |
| label_policy | complete_only: 모든 스타일 목표값이 있는 행만 사용 |
| pitch_baseline | median: 캐릭터별 학습 음원의 F0 중앙값들에 대한 중앙값 |
| grouping | character_text_and_audio_hash |
| no_pitch_policy | hold_out: 학습 피치 기준을 못 구하는 캐릭터는 보류 |
| max_speed_phonemes_per_second | 30.0: 초과 항목은 분할/통계 계산 전에 보류, 정확히 30은 유지 |

속도 30 기준은 사용자가 승인한 초기 품질 필터이며, 보편적인 발화 속도 한계라는 뜻은 아닙니다.
새 설정을 적용할 때는 기존 출력 ZIP을 보존하고 새 버전을 만듭니다.

## 동작

- 사람이 rejected로 표시한 행은 사용하지 않습니다. pending 허용 여부는 별도 설정입니다.
  pending을 쓰더라도 review_status를 accepted로 바꾸지 않습니다.
- 동일 캐릭터의 같은 대사(NFC 및 공백 정규화), 그리고 동일 음원 SHA256을 연결해
  전이적으로 묶습니다. 같은 그룹은 분할을 넘지 않습니다. SHA256은 캐릭터 간에도 묶습니다.
  문장부호나 표현이 다른 유사 대사의 의미적 중복 탐지는 하지 않습니다.
- 그룹을 seed 기반 해시 순서로 배치하고 캐릭터별 목표 비율과의 오차가 줄어들 때
  검증 쪽으로 이동합니다. 각 캐릭터의 학습 데이터는 남겨둡니다. 그룹 크기 때문에 비율은
  근사값이며 검증 데이터가 없는 캐릭터를 별도로 보고합니다. 어느 분할이 비면 중단합니다.
- 캐릭터 피치 기준에는 train의 유효 값만 사용합니다. 상대 피치는
  `12 * log2(발화 F0 중앙값 / 캐릭터 기준 F0)`입니다.
- speed/pitch/pause 각각의 train 유효 값으로 평균·모집단 표준편차를 구합니다.
  표준편차 0이면 scale=1로 기록하고, 학습 유효 값이 없으면 해당 목표를 마스킹합니다.
  결측값은 null이며 0이라는 정답으로 바꾸지 않습니다.
- 결과의 training_style에 감정 확률, 상대 피치, speed_z/pitch_z/pause_z 및 유효 마스크를
  추가합니다. 기존 pseudo_style은 그대로 남습니다.

`dataset-prepared-v1.zip`에는 train.jsonl, validation.jsonl, held-out.jsonl,
character-map.json, dataset-preparation.json, config.json, analyzer-config.json,
source-report.json이 들어갑니다. 기존 결과 ZIP은 덮어쓰지 않습니다.
학습 코드에서는 저장된 캐릭터 매핑과 통계를 그대로 사용해야 합니다.
