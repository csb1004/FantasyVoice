# Colab 전사 파일럿 결과 점검

입력: `pilot-v1.zip`. 원본 ZIP과 음성, 전사 기록은 수정하지 않았다.

## 실행 결과

- GPU: Tesla T4, 보고된 VRAM 14.56 GiB.
- Whisper large-v3 / openai-whisper 20250625, 한국어 전사, FP16.
- 선택 12개, 전사 상태 {'success': 12}. 실패/OOM 기록 없음.
- 이는 실행 성공의 증거이며 전사 정확도의 증거는 아니다.

## 데이터 범위

- Colab 목록: 796개 / 18개 캐릭터 / 약 36.75분.
- 로컬 전체: 25875개 / 89개 캐릭터 / 약 18.80시간.
- Colab 목록의 796개 모두 로컬 원본과 SHA256이 일치한다.
- 일부만 올린 것인지, 업로드/동기화 중 목록을 생성한 것인지 사용자 확인이 필요하다. 원인을 단정하지 않는다.

## 검수 상태

- reviews.jsonl이 ZIP에 없다. 12개 모두 전사 레코드상 pending이다.
- 이 점검에서는 음성을 청취하지 않았다. 자동 전사를 정답으로 확정하거나 수정하지 않았다.
- Adina의 “점성소래사는”, Bianca의 “그대의 기회를 시현할 기회를” 등은 문구상 검수 우선 후보이다. 실제 발음과 정답은 청취로 확인해야 한다.
- “흡”, “흥”, “으흠 으흐흠” 같은 짧은 전사는 비언어 발성 여부를 확인해야 한다. 텍스트만으로 자동 제외하지 않는다.

## 타임스탬프와 경로

- `Adela_KOR/Adela_selected2_1_ko.wav`: 실제 길이 3.213초, 마지막 ASR 구간 끝 4.000초. 정밀 정렬/휴지 정답으로 사용하지 않는다.
- `Aiden_KOR/Aiden_attack_27.wav`: 실제 길이 0.171초, 마지막 ASR 구간 끝 0.500초. 정밀 정렬/휴지 정답으로 사용하지 않는다.
- `Arda_KOR/Arda_attack4_01.wav`: 실제 길이 0.421초, 마지막 ASR 구간 끝 0.500초. 정밀 정렬/휴지 정답으로 사용하지 않는다.
- `Cathy_KOR/Cathy_selected2_1_ko.wav`: 실제 길이 5.375초, 마지막 ASR 구간 끝 6.560초. 정밀 정렬/휴지 정답으로 사용하지 않는다.
- Daniel 파일명은 Colab에서 한글 자모 분해형(NFD), 로컬에서는 완성형(NFC)이다. 해시는 같다. 파일 이동/결과 연결 시 정규화 충돌을 확인하고 해시로 검증해야 한다. 원본 경로를 일괄 변경하지 않았다.

## 전사 후보 목록

| 캐릭터 | 원본 파일 | 길이(초) | 자동 전사 |
|---|---|---:|---|
| Adela_KOR | `Adela_KOR/Adela_selected2_1_ko.wav` | 3.213 | 흥! 흠... |
| Adina_KOR | `Adina_KOR/Adina_joke_1_01.wav` | 14.897 | 점성소래사는 중첩해석을 강조하죠. 한 가지 방법으로만 운명을 해석하면 틀릴 수밖에 없다고요. 그러니까 어떻게 보면 끼워 맞추기라는 거예요. 제가 이런 말 했다고 이야기하지 마세요. |
| Aiden_KOR | `Aiden_KOR/Aiden_attack_27.wav` | 0.171 | 흡 |
| Alonso_KOR | `Alonso_KOR/Alonso_lobby_1_03.wav` | 13.323 | 아 잠깐만 좋아 작업을 시작해볼까 |
| Arda_KOR | `Arda_KOR/Arda_attack4_01.wav` | 0.421 | 흥! |
| Barbara_KOR | `Barbara_KOR/Barbara_joke_4_ko.wav` | 15.307 | 분자생물학자, 수학자, 공학자, 라지 사이즈 페퍼로니 피자 중 가장 이질적인 게 뭔지 알아, 수학자야, 왜냐하면 나머지는 한 가족을 먹여 살릴 수 있거든. |
| Bianca_KOR | `Bianca_KOR/Bianca_taunt_Emma_01.wav` | 12.685 | 마술! 마술! 나 마술 좋아해! 더 보여줘! 그대에게 그대의 기회를 시현할 기회를 선사하겠노라. |
| Bihyung_KOR | `Bihyung_KOR/Bihyung_PlaySkill1088200seq0_2_ko.wav` | 0.842 | 뚝딱! |
| Cathy_KOR | `Cathy_KOR/Cathy_selected2_1_ko.wav` | 5.375 | 으흠 으흐흠 |
| Celine_KOR | `Celine_KOR/Celine_joke_3.wav` | 13.179 | 컴포지션 포는 최고의 예술품이야, 가볍고 물에 안 녹는 데다 말랑말랑해서 아무 곳에나 붙여넣고 터뜨릴 수 있지, 먹지 마. |
| Chiara_KOR | `Chiara_KOR/chiara_selected_1_ko.wav` | 7.734 | 난 더 이상 구원받지 못할 거야. |
| Daniel_KOR | `Daniel_KOR/Daniel_taunt_수아_01.wav` | 12.389 | 시체 보존 방법에 대한 책도 있나, 흥, 있어도 안 읽어줄 거라고, 알았다, 그러면 나름의 방법을 써야겠군. |

## 다음 단계

1. 796개라는 입력 범위가 의도한 것인지 확인한다.
2. Colab 노트북의 검수 셀에서 12개를 청취하고 수정/판정을 저장한다.
3. reviews.jsonl을 포함해 결과를 전달한다. 이후 전체 전사 범위 및 선별 기준을 확정한다.
4. 학습 분할/스타일 추출 설정/조건 주입은 아직 미결정이므로 임의로 학습을 시작하지 않는다.
