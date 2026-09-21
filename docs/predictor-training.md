# 07 Style Predictor 실제 준비 학습

현재 대사로 감정 7개 확률과 속도·상대 피치·휴지를 예측하도록 KLUE BERT와
새 출력층을 함께 학습합니다. 캐릭터 ID·음성은 모델 입력에 넣지 않습니다.
음성 생성 및 MeloTTS 공동 학습은 이 단계에 포함되지 않습니다.

## Colab 실행

1. MyDrive/FantasyVoice에 dataset-prepared-v1.zip과 최신 fantasyvoice-pilot.zip을 둡니다.
2. **새 T4 GPU 런타임**에서 notebooks/07_train_predictor.ipynb를 실행합니다.
   scripts/colab_train_predictor.py 전체를 한 셀에 붙여넣어도 됩니다.
3. 학습 19,784개 / 검증 1,040개를 읽고 토큰 길이를 검사합니다. 문장을 자동 절단하지 않습니다.
4. 작은 배치로 실제 forward/backward가 통과하면 설정한 배치로 학습합니다.
5. 결과는 MyDrive/FantasyVoice/predictor-v1에 저장됩니다. 완료 후 검토용
   predictor-v1-report.zip을 다운로드합니다. 이 ZIP을 다음 결과 검토에 전달하세요.

07 의존성은 [predictor]로 분리했습니다. Whisper·Silero·MeCab·Melo를 설치하거나
불러오지 않습니다. Torch/Transformers 및 pinned KLUE BERT/tokenizer를 사용합니다.
CLS 표현을 직접 사용하므로 사용하지 않는 BERT pooler는 생성하지 않습니다.

## 조정 가능한 초기값

config/predictor.json 및 노트북 상단에서 배치 8, 누적 4, epoch 3,
검증 배치 16, BERT LR 2e-5, 출력층 LR 1e-4를 변경할 수 있습니다.
일반 업데이트의 유효 배치는 32이며, epoch 끝의 작은 배치도 실제 유효 라벨 수로 나눕니다.
그 밖의 설정은 config/predictor.json에 명시되어 있습니다.
노트북 상단에 노출된 값은 config/predictor.json의 같은 항목보다 우선합니다.

이 값들은 결과를 보고 조정할 실험 초기값입니다. 최적값이나 실제 T4 메모리 사용량을
검증한 값은 아닙니다. OOM이면 실행을 멈추고 메모리를 비운 새 런타임에서
예를 들어 배치 4/누적 8을 **새 OUTPUT 폴더**로 실험할 수 있습니다.
검증 중 OOM이라면 EVAL_BATCH_SIZE도 별도로 줄여야 합니다.

## 저장과 재개

- latest.pt: 모델/optimizer/scheduler/GradScaler/RNG/다음 위치를 포함한 재개 체크포인트.
- best.pt: 검증 총손실이 가장 낮았을 때의 학습 체크포인트.
- best-model.pt: 완료 후 내보낸 가중치·설정·데이터 통계 (optimizer 없이 후속 단계용).
- history.json: 평가 지점별 학습 window 손실과 validation 지표.
- validation-predictions.jsonl: 최적 모델의 검증 예측과 pseudo label 목표값/마스크.
- training-report.json: 각 손실, 원래 단위 오차, train 평균 예측 기준과 비교.
- tokenizer/: 저장한 tokenizer 파일.

100 optimizer updates마다 및 epoch 끝에 저장합니다. 일반 중단은 가능하면
마지막으로 완료된 업데이트 위치를 저장합니다. optimizer 업데이트 도중 중단되면
그 이전의 디스크 체크포인트를 유지합니다. 강제 런타임 종료 시 마지막 디스크 저장
이후의 작업은 반복할 수 있습니다. 체크포인트 파일은 임시 파일에 쓴 뒤 교체합니다.
전체 BERT와 Adam 상태를 보존하므로 모델 가중치만 저장할 때보다 저장 공간과 시간이 듭니다.

같은 설정·코드·패키지 버전·데이터·OUTPUT으로 재실행하면 latest.pt에서 재개합니다.
batch/epoch/LR 등을 바꿀 때는 OUTPUT을 predictor-v2 등으로 바꾸세요.
새 OUTPUT은 현재 구현상 원래 pretrained BERT에서 새로 시작합니다. epoch만 늘려
기존 optimizer/scheduler의 의미를 조용히 바꾸지 않습니다.
데이터 ZIP을 다시 압축한 경우에도 파일 해시가 바뀌므로 같은 실험의 재개 입력으로
대체하지 마세요. 재개용 .pt는 직접 생성한 신뢰하는 파일만 사용하세요.

FP16 gradient overflow는 GradScaler가 scale을 낮춘 뒤 같은 batch/RNG로 재시도합니다.
이때 optimizer update와 scheduler를 진행하지 않습니다. 8회 연속 실패하면 중단합니다.
이는 OOM에 대한 해결책은 아니며, 모델이나 배치를 자동으로 바꾸지 않습니다.

## 결과 해석과 범위

감정 soft CE와 3개 SmoothL1의 합으로 best를 선택합니다. 연속값 MAE는
표준화 단위와 원래 단위(음소/초, 반음, 휴지 비율) 모두 보고합니다.
비교 기준은 train 감정 평균 및 train 연속 목표 평균을 모든 검증 대사에 예측한 값입니다.
검증 라벨도 자동 분석기로 만든 pseudo label이므로 실제 사람 감정 정확도나
음성 생성 품질로 해석하지 않습니다. 예측된 연속값은 내보낼 때 임의로 클리핑하지 않습니다.

target_values 순서는 angry, disgusted, fearful, happy, neutral, sad, surprised,
speed_z, pitch_z, pause_z입니다. target_masks 순서는 emotion, speed, pitch, pause입니다.

로컬 검증: 실제 Transformers 작은 BERT의 학습/gradient, 중단·재개 동일성,
누적 배치 가중치, 결측 마스킹, 평가의 학습 상태/RNG 보존을 테스트합니다.
사전학습된 KLUE BERT 전체 모델도 실제 준비 데이터 2개로 CPU forward/backward를
통과했습니다. 전체 토큰화 결과 최대 길이는 train 71, validation 60이었습니다.
T4 FP16 학습의 메모리/속도 및 학습 품질은 사용자의 Colab 실행에서 확인합니다.
