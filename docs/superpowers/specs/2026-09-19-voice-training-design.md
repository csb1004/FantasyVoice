# FantasyVoice 음성 생성 학습 설계

작성일: 2026-09-19. 상태: 대화에서 승인된 요구사항을 정리한 설계 초안.
이 문서의 제안 사항은 구현 완료 또는 성능 검증 결과가 아니다.

## 1. 목적과 범위

한국어 대사에서 발화 스타일을 예측하고, 등록된 캐릭터의 목소리로 생성한다.
Style Predictor와 Conditional TTS를 한 파이프라인에서 준비 학습 후 공동 학습한다.
TTS 생성기 손실이 예측 스타일을 거쳐 Predictor까지 전달되는 경로가 있어야 한다.
강화학습 Actor-Critic은 구현하지 않는다.

첫 버전에서는 현재 대사만 사용한다. 원래 대화 순서가 없는 게임 음성을 임의로
이어 붙이거나 소설 문맥을 만들어 학습하지 않는다. 미래의 문맥 학습, RL,
voice conversion, 서비스 UI, SD Character 이미지는 현재 범위 밖이다.
감정 강도는 사용자의 명시적인 선택으로 첫 버전에서 보류한다.
분류 확신도, 음량, 각성도를 감정 강도로 대신 넣지 않는다.

## 2. 대화에서 확정된 결정

| 항목 | 결정 |
|---|---|
| 실행 환경 | Google Colab Pro / Pro+, 실제 할당 GPU는 실행 시 확인 |
| 원본 데이터 | 로컬 Voice 및 사용자가 동일 구조라고 설명한 MyDrive/Voice |
| 언어 | 한국어 중심 |
| 대사 자료 | 음성만 존재, 자동 전사로 준비 |
| ASR | Whisper large-v3, 일부 파일 검증 후 전체 적용 |
| TTS | MeloTTS 한국어 사전학습 모델을 PyTorch에서 확장 |
| Style Predictor | klue/bert-base + 감정 및 연속 스타일 출력층 |
| Predictor 입력 | 현재 대사만. 캐릭터 정보와 음성은 입력하지 않음 |
| 감정 연결 | 감정 확률로 학습 가능한 감정 임베딩을 가중합 |
| 화자 연결 | 캐릭터 ID별 학습 가능한 임베딩을 TTS에만 입력 |
| 신규 화자 | 첫 버전에서 지원하지 않음 |
| 감정 분석 | emotion2vec_plus_large, frozen, 샘플 검수 우선 |
| 감정 종류 | angry, disgusted, fearful, happy, neutral, sad, surprised |
| other / unknown | 검수 대기. 중립으로 자동 치환하지 않음 |
| 강도 | 첫 버전에서 예측·학습·제어하지 않음 |
| 피치 추출 | frozen CREPE / torchcrepe |
| 음성 구간 | frozen Silero VAD |
| 속도 | 정규화된 대사의 음소 수 / VAD 발화 구간 길이 합 |
| 피치 | 유성 구간 F0 중앙값의 캐릭터 기준 대비 반음 차이 |
| 휴지 | 첫 발화 시작~마지막 발화 종료 중 내부 비발화 시간 비율 |
| 스타일 시간 해상도 | 대사 전체의 요약값. 단어별 피치·휴지 위치 예측 없음 |
| 결측 | 해당 스타일 손실에서 마스킹, 임의의 정답 0으로 대체하지 않음 |
| 학습 순서 | pseudo style 기반 준비 학습 후 predicted style 기반 공동 학습 |
| 전사 오류 등 | 의심 파일은 검수 대기, 원본 보존 |

89개 폴더는 캐릭터 구분이며 실제 성우 89명을 의미하지 않는다.
로컬 헤더 조사에서 25,875개 중 25,859개를 읽었고 그 길이는 약 18.789시간이었다.
기본 WAV 리더로 읽지 못했던 16개는 이후 libsndfile로 읽었으며 FLOAT 형식이었다.
호환 리더 기준 25,875개 전부 헤더를 읽었고 총 길이는 약 18.8049시간이다.
세부 결과는 `docs/data-inventory.md`에 기록했다. 전체 디코딩/청취 검증 결과는 아니다.
Announce는 사용자가 선택적으로 허용했지만 첫 구현에는 필요성이 입증되지 않았다.
제안: Voice만으로 시작하고 Announce는 원본 그대로 보존한다.

## 3. 데이터 흐름

```text
Voice WAV -> 원본 목록/품질 검사 -> Whisper -> 전사 후보 -> 검수 상태
    |
    +-> frozen emotion2vec / CREPE / Silero -> pseudo style + 결측/출처
                                                       |
전사 text -> KLUE BERT -> predicted style ---------------+-> style loss
                              |
text + 캐릭터 embedding -------+-> MeloTTS 확장 -> 생성 음향/음성
                                                     |
reference audio -------------------------------------+-> TTS loss
                                                          |
                             공동 학습: TTS + Predictor 역전파

추론: text -> Predictor -> style -> TTS(text, character) -> WAV
```

분석기는 라벨 생성 시에만 실행하며 라벨을 캐시한다. ASR, 감정, 피치 분석기와
학습 모델을 동시에 GPU에 올릴 필요가 없다. 추론 API는 reference audio를 요구하지 않는다.

## 4. 스타일 의미와 한계

- 감정 분석 원본 9개 출력과 최종 학습용 7개 분포를 구분해 보존한다.
  7개 분포를 만드는 구체적인 정규화 및 검수 기준은 아래 결정 항목에 포함한다.
- 속도는 휴지를 제외한 조음 속도의 근사값이다. 음소 수 산출에는 실제 MeloTTS
  한국어 텍스트 전처리를 사용하며, 빈 토큰/구두점 등 비발음 토큰을 세지 않아야 한다.
- 피치 표현의 식은 `12 * log2(utterance_f0_median / character_baseline_f0)`이다.
  캐릭터 기준은 학습 분할에서만 구하고 추론 체크포인트에 저장한다.
- 휴지는 `(span_duration - sum(speech_interval_duration)) / span_duration`이다.
  speech span이 없거나 길이가 0이면 결측이다. 파일 앞뒤 무음은 분모에 포함하지 않는다.
- VAD 비발화 판정이 실제 언어적 휴지와 같지는 않다. 웃음, 숨소리, 기합을 점검한다.
- 요약값이 같아도 억양 곡선과 휴지 위치는 다를 수 있다. 참조 연기를 정확히
  재현한다고 보장할 수 없다. 같은 대사의 상이한 연기는 현재 입력만으로 구분되지 않는다.
- 캐릭터와 스타일을 별도 입력해도 통계적 독립이 자동으로 보장되지는 않는다.

## 5. 학습 손실과 단계

### 준비 학습

Predictor는 pseudo style target으로 학습하고, TTS는 pseudo style을 조건으로
데이터에 적응한다. 두 모듈을 별도 제품으로 분리하지 않고 동일 실행 흐름과
체크포인트 상태에서 관리한다. 준비 단계에서는 TTS loss가 Predictor로 전달되지 않는다.

### 공동 학습

TTS에 predicted style을 입력한다. 감정은 argmax 없이 확률 가중 임베딩을 사용한다.
속도·피치·휴지도 tensor로 전달하며 파일 저장/NumPy 변환/detach로 경로를 끊지 않는다.

```text
L_style = CE_soft(emotion_target, emotion_prediction)
        + SmoothL1(speed_prediction, speed_target)
        + SmoothL1(pitch_prediction, pitch_target)
        + SmoothL1(pause_prediction, pause_target)

L_joint_generator = L_Melo_generator + L_style
```

연속값은 학습 데이터 통계로 표준화하고 각 항목의 유효 표본에 대해서만 평균한다.
네 스타일 항목 초기 가중치는 각각 1이며 설정에 명시한다. 최적값이라는 주장은 하지 않는다.
한 배치에서 특정 라벨이 전부 결측이어도 NaN이 발생하거나 다른 항목까지 제외하면 안 된다.
MeloTTS 생성기 손실은 mel L1, duration, KL, adversarial, feature matching 구성을 유지한다.
공식 설정에 따른 duration discriminator 사용 여부와 계수는 실제 기반 설정을 확인해 명시한다.
판별기 손실은 생성기/Predictor 손실과 분리된 업데이트로 관리한다.
판별기 학습을 위한 의도적인 detach까지 무조건 제거하지 않는다.

## 6. MeloTTS 확장과 검증 경계

기존 MeloTTS는 이 프로젝트의 명시적인 감정/속도/피치/휴지 인터페이스를 제공하지 않는다.
새 조건 경로, 캐릭터 임베딩, 모델 로딩 호환 및 공동 trainer가 필요하다.
구체적인 조건 주입 위치와 기존 계층의 동결 범위는 아직 승인된 결정이 아니다.

학습의 posterior 경로가 실제 음성 정보를 사용하므로 재구성 품질만으로 성공을 판단하지 않는다.
다음 세 검증을 모두 수행한다.

1. 스타일 손실을 제외한 TTS 생성기 손실만 backward하여 Predictor의 유한한
   nonzero gradient를 확인한다. 감정 및 연속 스타일 경로를 각각 확인한다.
2. reference audio 없이 text와 character만으로 생성한다.
3. 같은 text/character/seed에서 스타일만 바꾸어 생성하고 변화 방향 및 음질을 평가한다.

미분 가능한 경로의 존재는 스타일 제어 품질의 증거가 아니다.
MeloTTS 기본 추론의 속도 조절 인자만 연결하고 이를 공동 학습으로 보고하지 않는다.

## 7. 파일 및 실행 구조 제안

```text
src/fantasyvoice/
  dataset/       inventory, manifest, transcripts, review, splits
  analyzers/     emotion, pitch, speech activity, feature calculation
  models/        style predictor, Melo adapter, character embeddings
  training/      losses, warmup/joint trainer, validation, checkpoint
  inference/     text/character -> waveform
config/          data, analyzer, model, training configuration
notebooks/       Colab 준비/샘플 전사/분석/학습/추론 진입점
tests/           데이터 무결성, 손실 마스크, gradient, 추론 계약 검증
docs/            설계, 실행 가이드, 파일럿 결과
```

제안 데이터 형식은 JSONL이다. 레코드에는 상대 오디오 경로, 캐릭터 ID, 원본 해시,
오디오 메타데이터, ASR 원문/수정문, 검수 상태/사유, split, analyzer 버전,
pseudo style 원본/가공값/유효 마스크를 둔다. pseudo label은 사람이 검수한 정답과 구분한다.
원본 WAV를 덮어쓰지 않고 처리된 사본과 캐시를 별도 디렉터리에 저장한다.
Drive 원본 경로와 출력 경로는 설정으로 주입한다. 로컬 Windows 경로를 Colab에 하드코딩하지 않는다.
Colab 재연결 시 optimizer, scheduler, scaler, RNG, 단계, step, 캐릭터 매핑,
정규화 통계, 모델/분석기 revision을 포함해 재개하도록 설계한다.

## 8. 구현 전 또는 파일럿에서 결정할 항목

아래 항목은 임의의 기본값으로 숨기지 않는다. 앞선 결정이 필요한 작업만 보류하며,
독립적인 원본 조사와 승인된 ASR 샘플 검증은 진행 가능한 범위로 분리한다.

| 항목 | 필요한 결정/근거 | 결정 시점 |
|---|---|---|
| 학습 manifest | JSONL 및 위 필드 구성 제안 | 데이터 코드 작성 전 |
| 오디오 전처리 | 기반 모델 sample rate에 맞춘 사본, mono 변환, trim/정규화 정책 | 전처리 코드 작성 전 |
| 데이터 분할 | 동일/중복 대사가 분할을 넘지 않도록 그룹화, 비율 및 seed | split 생성 전 |
| 감정 분포 | 7개 점수 재정규화, other/unknown 및 저확신 검수 기준 | 샘플 출력 확인 후 |
| 피치/VAD | 모델 크기, F0 범위, 유성 판단, 최소 발화/휴지 길이 | 샘플 음성 확인 후 |
| 캐릭터 피치 기준 | 학습 음성의 유효 F0 집계 방식 및 자료 부족 처리 | 스타일 추출 구현 전 |
| 조건 주입 | text encoder/flow/decoder/duration 경로 중 주입 위치와 gradient | Melo 구조 점검 후 구현 전 |
| 미세조정 범위 | BERT/TTS의 전체 또는 일부 계층 학습 여부 | GPU 메모리 조사 후 구현 전 |
| 학습 설정 | 학습률, 준비 단계 길이, batch, clip, stopping, 평가 간격 | 파일럿 설정 확정 전 |
| 성공 기준 | 청취 검수, 감정/속도/피치/휴지 제어 평가의 통과 기준 | 파일럿 평가 전 |

## 9. 구현 완료의 기준

- 샘플 ASR 결과와 검수용 오디오 연결을 제공하고 실패를 성공으로 기록하지 않는다.
- 검수 대기 데이터는 학습에서 제외되며 원본과 수정 이력이 보존된다.
- 분석기는 frozen이고 학습 optimizer에 포함되지 않는다.
- 데이터 분할 이후에만 정규화/캐릭터 기준 통계를 계산한다.
- TTS loss의 Predictor 역전파를 실제 모델에서 검증한다.
- 미등록 캐릭터와 지원하지 않는 intensity 입력은 명확히 거절한다.
- reference audio 없는 생성과 checkpoint 재개를 Colab에서 검증한다.
- 단위 테스트 통과, 작은 실제 GPU 학습 성공, 전체 학습 완료를 서로 구분해 보고한다.

## 10. 참고한 원문

- [MeloTTS 한국어 모델](https://huggingface.co/myshell-ai/MeloTTS-Korean)
- [MeloTTS 학습](https://github.com/myshell-ai/MeloTTS/blob/main/docs/training.md)
- [MeloTTS 모델 코드](https://github.com/myshell-ai/MeloTTS/blob/main/melo/models.py)
- [MeloTTS trainer](https://github.com/myshell-ai/MeloTTS/blob/main/melo/train.py)
- [Whisper](https://github.com/openai/whisper)
- [KLUE BERT](https://huggingface.co/klue/bert-base)
- [emotion2vec+ large](https://huggingface.co/emotion2vec/emotion2vec_plus_large)
- [torchcrepe](https://github.com/maxrmorrison/torchcrepe)
- [Silero VAD](https://github.com/snakers4/silero-vad)
