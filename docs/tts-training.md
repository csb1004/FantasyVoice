# 08 한국어 캐릭터 TTS 학습

07 Predictor 결과를 보존하고 MeloTTS를 캐릭터·스타일 조건에 적응시키는 단계다.
원본 Voice와 기존 train/validation 분할은 변경하지 않는다.

## 실행

1. 새 `dist/fantasyvoice-pilot.zip`을 Drive의 `MyDrive/FantasyVoice/`에 덮어쓴다.
2. `notebooks/08_train_tts.ipynb`를 Colab에서 열고 새 T4 런타임을 선택한다.
3. 기본 `STAGE = 'warmup'`으로 전체 셀을 실행한다.

필요한 Drive 파일:

```text
MyDrive/Voice/<캐릭터 폴더>/<음원.wav>
MyDrive/FantasyVoice/fantasyvoice-pilot.zip
MyDrive/FantasyVoice/dataset-prepared-v1.zip
MyDrive/FantasyVoice/predictor-v1/best-model.pt
```

07을 다시 학습할 필요는 없다. 보고서 ZIP은 가중치 파일을 대신할 수 없다.
기본 배치 1, 누적 4, 3 epoch, TTS learning rate 1e-4다. 설정을 바꾸면
`EXPERIMENT = 'v2'`처럼 새 이름을 사용한다. 동일 설정/데이터/코드/라이브러리의
같은 출력 폴더는 `latest.pt`에서 재개한다. v2는 자동으로 v1을 이어받지 않는다.

## 실행 중 보이는 단계

- 모델·한국어 frontend 로딩 및 데이터/가중치 출처 검사.
- 전체 20,824개 음원의 로컬 캐시 준비. 원본 SHA를 검사하고 44.1kHz mono 사본과
  텍스트 feature를 만든다. 무음 제거·음량 정규화는 하지 않는다.
- 실제 음원과 Predictor를 통한 TTS-only gradient 사전 점검.
- TTS 준비 학습: 자동 스타일 라벨 사용, Predictor 가중치는 변경하지 않음.
- 100 updates마다 샘플 평가/생성·저장, epoch 끝 전체 validation 평가.
- 최적 검증 mel 모델 내보내기와 text+character→Predictor→TTS 미리듣기 생성.

최초 캐시 준비는 Drive 음원을 모두 읽으므로 오래 걸릴 수 있다.
20개마다 진행 수를 출력한다. `/content`가 보존된 재실행에서는 준비한 캐시를
재사용하지만 새 런타임에서는 다시 만든다. 음원·feature 캐시에는 수십 GB의
로컬 디스크 공간이 필요할 수 있다. 캐시/다운로드는 Drive 원본을 덮어쓰지 않는다.

T4에서의 실제 전체 학습 시간·메모리 한도는 아직 측정하지 않았다.
배치 1 사전 점검도 가장 긴 발화나 optimizer 순간의 최대 메모리를 보장하지 않는다.
OOM이 나면 모델·데이터가 자동 변경되지 않고 오류가 표시된다.
NumPy 등 이전 패키지가 메모리에 남았다는 메시지가 나오면 런타임을 다시 시작한다.

## 결과와 재개

기본 결과는 `MyDrive/FantasyVoice/tts-warmup-v1/`에 저장된다.

| 파일 | 용도 |
|---|---|
| latest.pt | 세 optimizer와 scheduler/scaler/RNG/cursor를 포함한 중단 재개 |
| best-model.pt | 전체 validation 재구성 mel이 가장 낮은 generator, 매핑·설정·통계 |
| progress.json / history.json | 완료 update·epoch 및 검증 이력 |
| samples/step-… | 고정 대사/seed와 스타일 조건 비교 WAV·조건 JSON |
| predicted-style-preview.wav | 참조 음성 없이 Predictor 예측 스타일로 생성한 음성 |
| training-report.json | 학습 완료 상태와 검증 범위 |

`tts-warmup-v1-report.zip`은 마지막/최적 단계 샘플과 메타데이터만 포함한다.
큰 가중치는 Drive에 남긴다. 보고서와 생성 음성을 검수한 후 공동 학습으로 진행한다.
검증 mel 감소는 발음·캐릭터 유사도·스타일 제어의 청취 품질 통과를 뜻하지 않는다.

GAN의 세 optimizer는 같은 누적 창에서 gradient를 계산한 뒤 모두 유한할 때
함께 진행한다. 한 optimizer가 진행된 뒤 예외가 나면 이전의 완전한 디스크
checkpoint를 보존한다. 최대 저장 간격만큼의 작업을 재실행할 수 있다.
검증 중 끊겼으면 다음 업데이트 전에 해당 검증부터 다시 실행한다.

## 공동 학습을 시작할 때

같은 08 셀 상단에서 명시적으로 바꾼다.

```python
STAGE = 'joint'
EXPERIMENT = 'v1'
WARMUP_MODEL = BASE / 'tts-warmup-v1/best-model.pt'
```

출력은 별도 `tts-joint-v1`이다. 준비 학습 generator와 07의 Predictor 최적 가중치를
불러오며 새로운 optimizer/schedule을 시작한다. Predictor learning rate는 1e-5.
best-model에는 판별기가 없으므로 새로 초기화한다. 준비 학습 latest.pt를 지정하면
그 업데이트의 generator와 판별기 가중치를 함께 가져온다. optimizer까지 이어가는
동일 단계 재개와는 다르다.

공동 학습에서는 예측한 soft emotion과 연속값이 TTS에 들어가고,
TTS 생성기 손실 + 기존 4개 스타일 감독 손실로 Predictor와 TTS를 함께 학습한다.
캐릭터 ID는 Predictor에 입력하지 않는다.

## 출력 범위와 품질 해석

07의 raw 출력·구조를 바꾸지 않고 TTS 경계에서 속도와 휴지에 softplus 기반 범위
변환을 적용한다. 휴지는 0~1, 속도는 양의 범위로 보내며 피치는 유지한다.
관측된 음수 휴지 값에서도 gradient가 흐르지만, 극단적인 값에서는 수치적으로
포화할 수 있다. 휴지 0은 약 0.0139로 이동한다. raw/bounded 둘 다 샘플 JSON에 남긴다.

이는 실제 생성 음성의 측정 휴지 비율을 강제로 일치시키는 기능이 아니다.
같은 대사·캐릭터·seed에서 스타일 조건을 바꾼 음성을 비교해 실제 제어 효과를 평가한다.

## 로컬 검증

Melo source는 기존처럼 `--no-deps`로 설치하고 이 프로젝트의 한국어 전용 의존성을
사용한다. Melo의 전체 다국어/UI dependency 목록을 함께 설치하지 않는다.

```powershell
.venv/Scripts/python.exe -m pip install -e '.[tts,dev]'
.venv/Scripts/python.exe -m pip install --no-deps https://github.com/myshell-ai/MeloTTS/archive/209145371cff8fc3bd60d7be902ea69cbdb7965a.zip
.venv/Scripts/python.exe -m pytest -q
```

Windows에서 실제 한국어 frontend까지 실행하려면 eunjeon이 필요하다.
Colab Linux MeCab 의존성은 셀이 설치한다. 로컬 CPU 검증은 T4 학습 검증을 대신하지 않는다.
