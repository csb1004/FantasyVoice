# 10 공동 학습 모델로 음성 생성

09 완료 후 대사와 등록된 캐릭터 이름으로 음성을 생성한다. 추가 학습은 하지 않는다.
09의 같은 best-model.pt에서 Predictor와 TTS를 함께 불러온다.

## 실행

1. 최신 코드 ZIP을 `MyDrive/FantasyVoice/fantasyvoice-pilot.zip`에 교체한다.
2. `notebooks/10_generate_voice.ipynb`를 새 Colab 런타임에서 연다.
3. 첫 셀의 `JOINT_DIR`를 완료한 09 폴더로 맞추고 모델을 로딩한다.
4. 두 번째 셀의 `TEXT`, `CHARACTER`, `SEED`를 바꿔 실행한다.
5. 다른 대사를 생성할 때는 두 번째 셀만 다시 실행한다.

GPU와 CPU 모두 선택 가능하다. CPU는 느릴 수 있고 최초 실행에는 모델 다운로드가 필요하다.
원본 Voice 폴더, dataset ZIP, 음원 캐시, optimizer 체크포인트는 필요하지 않다.

필요한 파일:

```text
MyDrive/FantasyVoice/fantasyvoice-pilot.zip
MyDrive/FantasyVoice/tts-joint-v1/best-model.pt
MyDrive/FantasyVoice/tts-joint-v1/training-report.json
MyDrive/FantasyVoice/tts-joint-v1/run.json
MyDrive/FantasyVoice/predictor-v1/best-model.pt
```

07 파일은 학습 당시 Predictor의 정확한 구조·설정을 복원하고 출처를 확인하는 데 쓴다.
실제 생성에는 반드시 09에서 공동 학습된 Predictor 가중치로 교체한 뒤 사용한다.
08/09의 배치나 누적 설정을 10에 옮길 필요는 없다. 완료 검사는 09가 저장한 설정을 읽는다.

## 결과

생성할 때마다 `MyDrive/FantasyVoice/inference-v1/<시간-고유번호>/`에 저장한다.
이전 결과를 덮어쓰지 않으며 모델 가중치도 변경하지 않는다.

- `voice.wav`: 생성한 원본 float WAV, 44.1kHz. 음량을 정규화하지 않는다.
- `playback.wav`: 브라우저 청취용 PCM16 WAV. 피크가 1을 넘을 때만 재생용으로 감쇠한다.
- `generation.json`: 대사·캐릭터·seed·예측 스타일·범위 변환 값·모델 해시·재생 gain.
- 같은 이름의 ZIP: 위 결과를 묶어 다운로드한다. `DOWNLOAD=False`로 자동 다운로드를 끈다.

지원 캐릭터는 첫 셀에 출력된다. 등록되지 않은 캐릭터와 과도하게 긴 입력은
임의 대체나 자동 잘라내기 없이 오류를 표시한다. 참조 음성이나 임의 문맥을 입력하지 않는다.

대사별 발음, 캐릭터 유사도, 감정과 운율이 자연스러운지 청취한다.
생성 성공이나 검증 mel 최소라는 사실만으로 음질 합격을 뜻하지 않는다.
실제 사용자의 09 체크포인트를 이용한 생성은 해당 파일이 준비된 Colab에서 확인해야 한다.
