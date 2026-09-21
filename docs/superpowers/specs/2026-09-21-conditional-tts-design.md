# 08 Conditional TTS 구현안

상태: 2026-09-21 사용자 1번 선택으로 승인. 아래 설정은 초기 실험값이다.
기존 07 가중치, 원본 음원, 데이터 분할은 수정하지 않았다.

## 목적과 확정된 입력

한국어 대사와 캐릭터 ID로 스타일이 반영된 음성을 생성한다. 먼저 pseudo style로
MeloTTS를 준비 학습하고, 이후 Predictor의 predicted style로 공동 학습한다.
강화학습, 새 문맥 데이터, 감정 강도는 추가하지 않는다.
19,784 train / 1,040 validation과 기존 정규화·캐릭터 매핑을 그대로 사용한다.
Predictor는 대사만 입력받는다. 07 best-model.pt를 재사용한다.

## 제안 1: 기존 가중치를 보존하는 조건 어댑터 (권고)

### 출력 범위

07의 Linear 출력과 저장 형식, 원래 스타일 감독 손실은 유지한다.
TTS에 전달하는 지점에서 FP32 tensor로 역표준화하고 범위를 처리한 뒤
기존 train 통계로 다시 표준화한다. NumPy/detach/argmax는 쓰지 않는다.

- 휴지 비율 r: `t * (softplus(r/t) - softplus((r-1)/t))`, t=0.02.
  0~1 범위의 부드러운 제한이다. 수치적으로 안정적인 동등식으로 구현하고
  극단값, 경계, 도함수를 시험한다. 극단값에서는 gradient가 소실될 수 있다.
- 속도 s: `t * softplus(s/t)`, t=0.1 음소/초. 양의 값으로 제한한다.
- 캐릭터 상대 피치: 기존 반음 값을 유지한다. 추가 임의 상한은 두지 않는다.
- pseudo style에도 같은 변환을 적용하여 준비/공동 학습의 조건 표현을 맞춘다.
- 원본 값과 변환 값은 둘 다 평가 보고서에 남긴다.

이는 예측 정확도 개선 자체가 아니다. 예를 들어 원래 휴지 0은 약 0.0139가 된다.
경계 부근을 조금 이동시키는 대신 음수 예측에서도 학습 가능한 경로를 얻는 선택이다.
출력 범위를 고치는 것과 실제 생성 음성의 휴지 비율을 보장하는 것은 구분한다.

### 조건 경로

Melo의 기존 256차원 g 조건을 확장한다.

`g = character_embedding[id] + emotion_probabilities @ emotion_embedding + Linear(style_z)`

- 캐릭터 임베딩 89x256: 한국어 KR 임베딩으로 각 행을 초기화하고 학습한다.
- 감정 임베딩 7x256과 연속값 Linear(3,256): 작은 난수(std=0.01), bias=0.
  처음부터 style gradient를 막는 완전한 영행렬 초기화는 하지 않는다.
- 같은 g를 text encoder, posterior encoder, flow, waveform decoder,
  deterministic/stochastic duration predictor에 전달한다.
- duration predictor의 기존 x.detach는 유지하되 g.detach는 제거해 조건의
  역전파를 허용한다. 판별기 업데이트의 의도적인 detach는 유지한다.
- 학습/추론에서 같은 조건 어댑터를 사용한다. 전역 스타일 조건이며
  특정 위치의 쉼이나 정확한 F0 곡선을 지정하는 인터페이스는 아니다.
- 사전학습 로딩 시 기존 가중치 누락은 새 조건 파라미터에만 허용한다.
  임의 strict=False로 다른 불일치를 숨기지 않는다.

### 음원과 텍스트

- 원본 SHA를 확인하고 별도 로컬 캐시에 mono float waveform으로 디코딩한다.
- 다채널은 채널 평균, sample rate는 한국어 모델 기준 44,100 Hz로 변환한다.
- 앞뒤/내부 무음 trim, loudness/peak normalization을 하지 않는다.
- 비유한 값·무음·잘못된 파일은 이유와 함께 중단/보고하며 조용히 누락하지 않는다.
- 전체 발화를 정렬에 사용하고 decoder 학습 segment는 Melo 기본 16,384 samples.
  긴 발화를 임의 잘라 새 대사로 만들지 않는다.
- 기존 pinned Melo 한국어 frontend를 사용한다. 토큰 길이·symbol ID·BERT
  feature 구성은 공식 한국어 경로와 대조하며 임의 0 feature로 대체하지 않는다.
- 캐시 키는 원본 SHA, 전사, frontend revision, 전처리 설정을 포함한다.

### 학습 범위와 초기 실험값

08은 실제 TTS 준비 학습과 참조 음성 없는 평가 생성을 제공한다.
Predictor는 준비 학습 optimizer에 포함하지 않는다. 별도 명시적인 공동 학습
모드에서만 Predictor를 로드하고 07 최적 가중치부터 함께 업데이트한다.
준비 학습 종료만으로 공동 학습을 자동 시작하지 않는다.

TTS 생성기 전체와 새 조건층을 미세조정하고 waveform/duration 판별기를 학습한다.
손실은 공식 Melo mel(45), KL(1), duration, adversarial, feature matching을 유지한다.
기존 추론용 generator 가중치에 판별기가 없으면 판별기는 새로 초기화하고 기록한다.

초기 실험 제안: seed 42, micro batch 1, 누적 4, 3 epochs,
TTS 생성기·판별기 AdamW lr=1e-4, betas=(0.8,0.99), eps=1e-9,
weight_decay=0.01, epoch별 lr_decay=0.999875, gradient norm clip=1.
새 조건층도 같은 lr. 100 optimizer updates마다 저장·고정 검증 샘플 생성,
epoch 끝에 전체 validation 손실 평가. FP16 autocast를 사용하되
정렬·분포·손실 계산은 FP32로 보호한다. 모두 실측 전의 조정 가능한 값이다.

공동 학습에서는 Predictor lr=1e-5, 기존 스타일 손실 네 항목 가중치 1을
TTS 생성기 손실에 더한다. 별도 출력 디렉터리와 optimizer/schedule을 사용한다.
단계 변경은 warm start이며 같은 단계의 중단 재개와 구분한다.

T4 메모리 수용 여부는 실제 GPU preflight로 확인해야 한다. OOM 시 모델이나
데이터를 자동 변경하지 않고 사용 중인 설정과 실패를 보고한다.

### 저장과 평가

- 데이터/설정/모델 revision 해시와 캐릭터·감정 매핑·통계를 저장한다.
- generator, 모든 discriminator, Predictor(공동 학습), 각 optimizer/scheduler,
  scaler, RNG, sampler/cursor와 단계 정보를 저장한다.
- 여러 optimizer 업데이트 도중 중단된 상태는 정상 checkpoint로 덮어쓰지 않는다.
- 고정 text/character/seed 생성과 감정·속도·피치·휴지 각각의 조건 변경 비교를 제공한다.
- GAN 총손실만으로 청취 품질 최고라고 선언하지 않는다. 검증 mel 최소 checkpoint와
  latest를 구분하고 생성 샘플을 함께 보존한다.
- 사람이 듣기 전에는 발음·캐릭터 유사도·스타일 제어의 품질 통과를 선언하지 않는다.

## 대안

2. TTS 추론에서만 hard clamp: 값 보존은 쉽지만 범위 밖의 derivative가 0이므로
   현재 음수 휴지 예측들의 공동 학습 경로에 불리하다.
3. Predictor를 sigmoid/logit 기반 출력으로 바꾸고 재학습: 범위를 모델 내부에서
   처리할 수 있지만 기존 07 출력 의미와 가중치 호환·감독 손실 설계를 바꿔야 한다.

## 구현 검증 기준

1. 범위 변환의 유한 값/gradient/경계 및 raw metadata 보존.
2. 기존 Predictor checkpoint 호환, 미등록 캐릭터 거절.
3. 작은 실제 Melo 모델에서 TTS 손실만으로 emotion과 각 연속값 및
   Predictor 파라미터까지 finite nonzero gradient 도달 확인.
4. 모든 discriminator와 accumulation 마지막 부분 배치를 포함한 재개 검증.
5. 실제 한국어 사전학습 가중치 로딩 및 reference audio 없는 WAV 생성.
6. 작은 실제 Colab GPU 업데이트와 전체 학습을 별도로 보고.

## 확인한 공식 자료

- https://github.com/myshell-ai/MeloTTS/blob/209145371cff8fc3bd60d7be902ea69cbdb7965a/melo/models.py
- https://github.com/myshell-ai/MeloTTS/blob/209145371cff8fc3bd60d7be902ea69cbdb7965a/melo/train.py
- https://github.com/myshell-ai/MeloTTS/blob/209145371cff8fc3bd60d7be902ea69cbdb7965a/melo/configs/config.json
- https://huggingface.co/myshell-ai/MeloTTS-Korean/blob/main/config.json

실행 코드에서는 한국어 모델의 실제 revision과 artifact SHA도 고정해야 한다.
공식 추론 config에는 학습 설정 일부가 없으므로 위 optimizer 등은 프로젝트 제안이다.
