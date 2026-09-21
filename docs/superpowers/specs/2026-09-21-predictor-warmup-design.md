# 07 Style Predictor 준비 학습 제안

상태: 2026-09-21 사용자 1번 승인. 배치·epoch 등은 결과를 보고 조정할 초기 실험값이다.
실제 GPU 학습 완료를 뜻하지 않는다.

## 이미 합의된 범위

- 한국어 현재 대사만 입력하는 klue/bert-base 기반 Style Predictor.
- 캐릭터 ID나 reference 음성을 Predictor에 넣지 않는다.
- 감정 7개 확률과 속도·상대 피치·휴지 3개 연속값을 출력한다.
- 감정은 soft-target cross entropy, 연속값은 train 통계로 표준화한 SmoothL1.
- 네 손실의 초기 가중치는 각각 1. 결측 라벨은 마스킹한다.
- 이후 MeloTTS 연결과 공동 학습에 재사용한다. 여기서는 TTS를 학습하거나 생성하지 않는다.
- 입력은 dataset-prepared-v1.zip: train 19,784개 / validation 1,040개.
  승인된 분할과 통계를 그대로 읽으며 다시 분할하지 않는다.

## 이번에 승인할 구현 제안

### 구조

- BERT 마지막 hidden state의 첫 CLS 토큰을 사용한다.
- dropout 0.1 후 감정 Linear(hidden_size, 7), 연속값 Linear(hidden_size, 3).
- 감정 순서는 angry, disgusted, fearful, happy, neutral, sad, surprised로 고정한다.
- 출력은 emotion_logits와 speed_z/pitch_z/pause_z. 학습 중 softmax 확률을
  argmax나 detach로 바꾸지 않는다. TTS 연결은 이후 별도로 구현한다.
- BERT 전체와 새 출력층을 함께 미세조정한다.

### T4 초기 학습 설정 (성능/메모리 실측 전)

| 항목 | 제안 |
|---|---|
| seed | 42 |
| 최대 토큰 | 256, 초과 시 조용히 자르지 않고 사전 점검에서 중단 |
| micro batch / 누적 | 8 / 4 (일반적인 업데이트의 유효 batch 32) |
| epoch | 3 |
| optimizer | AdamW, betas=(0.9,0.999), eps=1e-8 |
| 학습률 | BERT 2e-5, 새 출력층 1e-4 |
| weight decay | 0.01, bias와 LayerNorm에는 0 |
| scheduler | 전체 업데이트 중 처음 10% 선형 warmup, 이후 선형 감소 |
| precision | CUDA FP16 autocast + GradScaler, 손실 계산 FP32 |
| gradient clip | 전체 gradient norm 1.0 |
| 평가 | 100 optimizer updates마다 및 epoch 끝, validation batch 16 |
| 저장 | 100 optimizer updates마다 및 epoch 끝, latest와 validation best |

최초 학습 셀은 의존성 import, tokenizer와 모델 로딩, 길이 및 대상값 점검,
실제 작은 batch의 forward/backward/finite-gradient 검사를 먼저 수행한다.
메모리가 부족해도 모델/학습 범위를 자동 교체하지 않고 설정 조정 필요를 표시한다.

### 재개와 산출물

- tokenizer/model revision 고정, 입력 ZIP 해시·config·통계·라벨 순서 저장.
- 모델, optimizer, scheduler, scaler, RNG, epoch, 다음 batch 위치를 저장한다.
- 체크포인트는 gradient 누적 경계에서만 저장한다. 일반 중단에도 완료된 경계를 보존한다.
- 설정이나 입력이 다른 체크포인트는 이어 학습하지 않는다.
- 체크포인트 latest/best, 손실 이력, validation 예측 JSONL, 평가 요약 ZIP을 Drive에 저장한다.
- 감정 soft CE와 각 연속값 오차 및 상수 예측 기준과의 비교를 보고한다.
  pseudo label 평가이므로 실제 감정 정확도나 음성 생성 품질로 표현하지 않는다.

## 비교한 대안

1. 위와 같이 Predictor 준비 학습부터 구현: TTS 구조 결정과 독립적이고 기존 라벨을 바로 사용.
2. BERT를 고정하고 출력층만 학습: 메모리/연산 부담이 작지만 BERT의 표현 자체는 적응하지 않음.
3. 바로 TTS와 공동 학습 구현: 조건 주입 위치, TTS 동결 범위, 음원 전처리 및
   별도의 TTS optimizer/batch 설정을 먼저 정해야 하므로 현재 미정 사항이 더 많음.

권고는 1번이다. 07은 실제 Predictor 학습 단계이며 추가 데이터 검증 단계가 아니다.

## 구현 시 검증 기준

- 캐릭터와 음원이 Predictor 입력으로 들어가지 않는다.
- 모든 목표값 누락 시 finite zero loss와 올바른 마스킹을 보장한다.
- BERT와 모든 출력층에 finite gradient가 도달한다.
- validation에서 weight/optimizer/RNG 학습 상태를 잘못 변경하지 않는다.
- 불완전한 마지막 누적 batch도 실제 표본 수에 맞게 가중한다.
- 재개한 다음 업데이트와 연속 실행 결과를 작은 CPU fixture로 비교한다.
- 잘못된 데이터/설정의 checkpoint 재사용을 거부한다.
- 로컬 테스트와 실제 KLUE BERT T4 학습 검증 결과를 별도로 보고한다.

## 참고

- KLUE BERT: https://huggingface.co/klue/bert-base
- 기존 전체 설계: 2026-09-19-voice-training-design.md
- MeloTTS 조건/gradient 검토 대상:
  https://github.com/myshell-ai/MeloTTS/blob/209145371cff8fc3bd60d7be902ea69cbdb7965a/melo/models.py
