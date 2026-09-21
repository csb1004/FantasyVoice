# 09 Predictor + TTS 공동 학습

08을 기본 warmup 모드로 완료한 다음 실행한다. 07의 최적 Predictor와
08의 최적 TTS를 불러와 예측 스타일을 통해 두 모델을 함께 학습한다.
기존에 승인한 공동 학습 기능을 전용 노트북으로 제공한다.

## 실행

1. 최신 `fantasyvoice-pilot.zip`을 `MyDrive/FantasyVoice/`에 교체한다.
2. `notebooks/09_joint_tts.ipynb`를 새 GPU Colab 런타임에서 실행한다.
3. 기본 설정은 batch 1, accumulation 4, 3 epochs, TTS lr 1e-4, Predictor lr 1e-5다.

필요한 입력:

```text
MyDrive/Voice/<캐릭터>/<음원.wav>
MyDrive/FantasyVoice/fantasyvoice-pilot.zip
MyDrive/FantasyVoice/dataset-prepared-v1.zip
MyDrive/FantasyVoice/predictor-v1/best-model.pt
MyDrive/FantasyVoice/tts-warmup-v1/best-model.pt
MyDrive/FantasyVoice/tts-warmup-v1/training-report.json
MyDrive/FantasyVoice/tts-warmup-v1/run.json
```

08에서 다른 실험 이름을 썼다면 `WARMUP_MODEL`을 해당 폴더의 `best-model.pt`로
변경한다. 같은 폴더의 완료 보고서와 실행 정보도 필요하다.
09는 미완료 08 결과나 다른 실행의 보고서/가중치가 섞인 경우 시작하지 않는다.
데이터셋·캐릭터 매핑·정규화 통계·07 가중치·모델/frontend 설정을 대조한다.

## 학습과 결과

- 08에서는 자동 스타일 라벨을 TTS에 입력했다. 09에서는 현재 대사만으로
  Predictor가 예측한 감정 확률·속도·피치·휴지를 TTS에 전달한다.
- TTS 손실과 기존 스타일 감독 손실로 Predictor와 TTS를 함께 학습한다.
- 08 best-model에는 판별기가 없어 두 판별기는 새로 초기화한다.
  optimizer/schedule도 새로운 공동 학습 실험으로 시작한다.
- 기존 07/08 결과는 변경하지 않고 `MyDrive/FantasyVoice/tts-joint-v1/`에 저장한다.
- `latest.pt`는 공동 학습 재개용이다. `best-model.pt`는 최적 TTS와 Predictor를 함께 포함한다.
- `predicted-style-preview.wav`와 `samples/`에서 결과를 청취한다.
- 완료 시 `MyDrive/FantasyVoice/tts-joint-v1-report.zip`을 만들고 다운로드한다.

epoch·batch·학습률을 바꾸면 `EXPERIMENT = 'v2'`처럼 새 이름으로 실행한다.
같은 설정·코드·의존성의 중단 재개는 같은 폴더에서 이어간다.
이전에 08의 joint 모드를 사용했다면 09는 실행 진입점이 달라 기존 joint 폴더를
그대로 재개하지 않는다. 원래 코드로 재개하거나 09에 새 실험 이름을 사용한다.

09는 가벼운 결과 정리 셀이 아니라 실제 공동 학습이다. Predictor까지 학습하므로
08보다 GPU 메모리 부담이 커질 수 있다. T4 전체 공동 학습은 아직 미검증이다.
OOM이 나도 모델·데이터를 자동 변경하지 않는다.

## 08 resumable-cache-v2 캐시 재사용

사용자가 수정한 `08_train_tts_resumable_cache_v2.ipynb`의 형식과 호환된다.
`MyDrive/FantasyVoice/tts-cache-checkpoints/cache-part-*.tar.gz`를 순서대로
`/content/fantasyvoice-tts-cache`에 복원한 뒤 기존 해시 기반 캐시를 사용한다.
08 원본과 모델·데이터·frontend·패키지 버전이 같으면 다시 음원을 읽거나
동일 텍스트 feature를 계산하지 않는다. 버전·전사 등이 달라 키가 바뀐 항목은 새로 만든다.
캐시가 있어도 tokenizer/frontend 모델 초기화와 캐시 목록 순회는 실행한다.

09에서도 `CACHE_CHECKPOINT_EVERY = 500`개 처리마다 새 파일만 증분 저장하며,
정상 종료와 일반 중단 시 남은 완성 파일도 저장을 시도한다. 런타임 강제 종료 시에는
마지막으로 저장된 파트까지만 복구할 수 있다. `.partial` 업로드는 복원 대상에서 제외한다.
한 체크포인트 폴더는 한 런타임에서만 기록한다.

저장된 파트에는 `audio/<hash>.pt`, `text/<hash>.pt`만 허용하며 링크나 캐시 폴더 밖
경로를 거부한다. 사용자의 08 변형은 참고용 파일로 보존했고 원래 08 진입점·src·config는
수정하지 않았다. 이미 이전 09로 학습 중이라면 그 코드를 유지해 재개해야 한다.

완료 보고서가 있다고 해서 청취 품질을 통과한 것은 아니다. 같은 대사·캐릭터·seed의
스타일 비교 샘플로 발음, 캐릭터 유사도, 스타일 반응을 검수한다.

## 완료 후

`notebooks/10_generate_voice.ipynb`에서 임의 대사와 등록된 캐릭터로 음성을 생성할 수 있다.
모델을 한 번 로딩한 후 생성 셀만 반복 실행한다. [10 안내](inference.md)를 참고한다.
