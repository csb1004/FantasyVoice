# 스타일 라벨 생성

전체 전사 → `03_after_dataset.ipynb` 후속 점검 → **새 GPU 런타임**에서
`04_style_labels.ipynb` 순서로 실행합니다. 진행 중인 전사 런타임에서는 실행하지 않습니다.
현재 T4를 선택할 수 있으나 실제 Colab GPU 호환성·속도는 첫 실행에서 확인해야 합니다.

최신 `dist/fantasyvoice-pilot.zip`을 MyDrive/FantasyVoice에 업로드합니다.
실행 코드는 `scripts/colab_style_labels.py`에도 있습니다.
입력은 후속 점검으로 생성된 `MyDrive/FantasyVoice/dataset-report.zip` 안의
`dataset-v1/candidates.jsonl`입니다. 수정 대사를 포함한 별도 목록은
`CANDIDATES_OVERRIDE`에 지정할 수 있습니다. 결과는 `MyDrive/FantasyVoice/style-v1`입니다.
12개 제한 없이 후보 전체를 처리합니다. 검수 미완료 자료도 미리 분석하되 승인으로 바꾸지 않습니다.

## 저장 형식

입력의 `audio_path`, `character_id`, `sha256`, `text`, `review_status`, `split` 등을
유지하고 아래 필드를 추가합니다. `key`는 전사 식별자 그대로입니다.

| 필드 | 의미 |
|---|---|
| `style_key` | 음원 경로·해시·대사·분석 설정 및 구현 버전의 식별자 |
| `style_status` | generated / partial / error / excluded / pending |
| `pseudo_style.emotion.raw_probabilities` | 원래 9개 감정 확률, `<unk>` 이름은 unknown으로 통일 |
| `pseudo_style.emotion.probabilities` | 합계 1의 7개 감정 확률; other/unknown이 최대이면 null |
| `pseudo_style.speed_phonemes_per_second` | Melo 한국어 음소 수 / Silero 발화 시간 합계 |
| `pseudo_style.pitch_f0_median_hz` | VAD 발화 구간의 신뢰 가능한 CREPE F0 중앙값 |
| `pseudo_style.pause_ratio` | 첫 발화 시작~마지막 발화 종료 중 내부 비발화 비율 |
| `pseudo_style.pitch_semitones` | null: 학습 분할 내 캐릭터 기준값을 정한 뒤 계산 |
| `pseudo_style.valid_targets` | 각 목표값 사용 가능 여부; normalized pitch는 현재 false |
| `pseudo_style.issues` | 누락 목표값, 발화 미검출, other/unknown 등 확인 이유 |

감정 강도는 만들지 않습니다. 원시 피치를 상대 피치 대신 학습에 투입하지 않습니다.
스타일 분석 성공은 대사 정확도나 사람의 검수 통과를 뜻하지 않습니다.
문장부호·경계 토큰은 음소 수에서 제외하며, 미지 토큰이나 지원하지 않는 음소가
있으면 속도 값은 null입니다. 입력 음원은 해시 확인 후 mono/16kHz로 메모리에서만 변환합니다.

## 임시 추출 설정과 재개

`config/style.json`의 VAD 임계값·최소 구간, F0 범위 50~1100Hz,
10ms hop, periodicity 0.21 등은 **조정 가능한 초기 추출 설정**입니다.
품질 합격 기준으로 확정한 값이 아닙니다. VAD padding은 0ms로 두어 속도·휴지
계산에 인위적으로 붙인 여백이 들어가지 않게 했습니다. 모델 revision은 고정합니다.
패키지 실제 버전과 구현 해시도 출력 `config.json`에 기록합니다.

- 20개 처리마다 `results/*.jsonl` 체크포인트 저장. 강제 런타임 종료 시 최대 19개 재계산.
- 같은 코드를 같은 출력 폴더로 재실행하면 generated/partial/excluded를 재사용하고 error를 재시도합니다.
- 대사 수정 시 해당 항목의 키가 바뀌므로 다시 분석합니다. 검수 상태만 바뀌면 분석을 재사용합니다.
- 설정·분석 코드·패키지 버전 변경 시 새 출력 폴더(style-v2 등)를 지정합니다.
- partial은 모델이 값을 만들지 못한 결과입니다. 같은 설정에서 자동 반복하지 않으며
  `style-needs-review.jsonl`에서 확인 후 대사/설정을 수정합니다.
- `style-labels.jsonl`은 종료 또는 정상 중단 시 전체 내보내기입니다. 강제 종료 시
  체크포인트가 더 최신일 수 있으므로 재실행으로 재생성합니다. 한 출력 폴더당 실행은 하나만 사용합니다.

로컬 테스트는 계산·파일 처리·캐시/재개 동작을 검증합니다. 실제 분석기 가중치를
다운로드한 Colab GPU 실행과 감정 정확도 평가는 별도입니다. 시작 시 첫 음원 분석에
실패하면 전체 반복에 들어가기 전에 중단하므로 오류 메시지를 확인할 수 있습니다.

### Silero import 시 onnxruntime 누락

Silero VAD 6.2.2는 패키지 import 중 sequence_vad를 통해 onnxruntime을 불러옵니다.
PyTorch VAD를 사용해도 import 의존성이 필요해 프로젝트 style 의존성에
`onnxruntime==1.24.4`를 명시했습니다. 기존 Colab에서는 새 셀에
`%pip install -q onnxruntime==1.24.4`를 실행한 다음 04 실행 셀을 다시 실행하세요.
모델 초기화 중 발생한 이 오류는 음원 분석 시작 전이므로 결과를 삭제할 필요가 없습니다.

### Python 3.13에서 g2pkk의 NoneType.pos 오류

g2pkk는 MeCab 초기화 실패를 출력만 하고 넘어갑니다. Python 3.13 Linux에는
원래 python-mecab-ko 1.3.7 wheel이 없어 별도 호환 빌드
`python-mecab-ko-py313==1.3.7.dev0`와 `python-mecab-ko-dic==2.1.1.post2`를 명시했습니다.
호환 빌드의 Python wrapper는 원본 1.3.7과 버전 문자열 외에는 동일함을 확인했습니다.
실제 Linux 바이너리 실행은 Colab에서 확인해야 합니다.
재시도 시 Melo의 `korean.g2p_kr`에 실패한 객체가 남을 수 있어 새 코드가 이를 초기화합니다.
기존 실행 셀을 유지할 경우, 호환 패키지 설치 후 `from melo.text import korean`과
`korean.g2p_kr = None`을 실행하고 04 셀을 재실행하세요.

분석기 API 근거: [emotion2vec](https://huggingface.co/emotion2vec/emotion2vec_plus_large),
[torchcrepe](https://github.com/maxrmorrison/torchcrepe),
[Silero VAD](https://github.com/snakers4/silero-vad),
[Melo 한국어 전처리](https://github.com/myshell-ai/MeloTTS/blob/209145371cff8fc3bd60d7be902ea69cbdb7965a/melo/text/korean.py).
