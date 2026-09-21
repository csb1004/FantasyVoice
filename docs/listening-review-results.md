# 12개 청취 검수 결과

- 사용자 파일: D:/DATA/Downloads/reviews.jsonl. 원본 수정 없음.
- 저장 이벤트 17건, 고유 음성 12개. 파일 순서상 마지막 저장을 현재 판정으로 집계.
- 현재 판정: accepted 5 / rejected 1 / pending 6. pending은 저장했더라도 합격으로 간주하지 않음.
- 기존 run.json의 설정과 로컬 상대 경로/해시로 key를 재계산해 12개 모두 원본과 연결.
- 최신 전체 transcripts.jsonl은 받지 않았으므로 전체 전사의 WER/CER는 계산하지 않음.
- 12개 중 10개가 0.5초 이하. accepted 5개도 모두 이 길이 조건에 해당.
- 판정 기록은 유지하되 새로운 길이 정책에서 제외. 길이 통과한 2개는 rejected 1, pending 1.

## 관찰

- 무음 Adela 파일의 검수 텍스트에 “자막 제공 및 자막 제공 및 광고를 포함하고 있습니다.”가 남아 있음. 사용자가 note에 무음이라고 기록했고 로컬 원본도 완전 무음으로 확인된 사례.
- Adriana 파일은 수정문이 “추운 건 싫어... 그렇지?”이고 판정은 rejected. 전사 오류만 수정한 것인지, 음성 자체가 부적합한 것인지 확인 필요. accepted로 임의 변경하지 않음.
- 비언어 발성에 치우친 표본이므로 일반 대사 전체 성능으로 확대 해석하지 않음.

## 길이 정책 적용

- 전체 25,875개 중 0.5초 이하 1,121개 제외, 길이 통과 24,754개.
- 경계는 엄격히 duration > 0.5초. 원본 삭제 없음.
- 현재 구현된 파일럿 후보 선정과 Colab 전사 입력에 적용. 이후 학습 manifest도 동일 정책을 적용해야 함.
- 길이 통과는 최종 학습 합격 판정이 아님. 0.5초를 넘는 기합/웃음 등은 별도 검수.

| 파일 | 초 | 마지막 판정 | 수정문 | 길이 제외 |
|---|---:|---|---|---|
| `Abigail_KOR/Abigail_attack1_07.wav` | 0.356 | pending | 하! | True |
| `Adela_KOR/Adela_PlaySkill1024200seq0_4_ko.wav` | 0.063 | pending | 자막 제공 및 자막 제공 및 광고를 포함하고 있습니다. | True |
| `Adina_KOR/Adina_attack3_06.wav` | 0.164 | accepted | 히! | True |
| `Adriana_KOR/adriana_firstMove_1_ko.wav` | 3.877 | rejected | 추운 건 싫어... 그렇지? | False |
| `Aiden_KOR/Aiden_attack_27.wav` | 0.171 | accepted | 흡 | True |
| `Alex_KOR/Alex_attack_17.wav` | 0.323 | pending | 흣! | True |
| `Alonso_KOR/Alonso_attack1_09.wav` | 0.289 | pending | 흣 | True |
| `Arda_KOR/Arda_attack4_01.wav` | 0.421 | pending | 흥! | True |
| `Aya_KOR/aya_attack_1_ko.wav` | 0.384 | accepted | 핫! | True |
| `Barbara_KOR/Barbara_attack_2_ko.wav` | 0.312 | accepted | 흡 | True |
| `Bianca_KOR/Bianca_attack_05.wav` | 0.325 | accepted | 흫 | True |
| `Bihyung_KOR/Bihyung_PlaySkill1088200seq0_2_ko.wav` | 0.842 | pending | 뚝딱! | False |

## 사용자 확인 반영

- 사용자가 Adriana 음성은 정상이며 전사만 틀렸다고 확인했다.
- 수정문 “추운 건 싫어... 그렇지?”를 유지하고 accepted 이벤트를 추가했다.
- 현재 판정: accepted 6 / rejected 0 / pending 6. 이 중 길이 기준도 통과한 accepted는 1개다.
- 기존 17개 이벤트는 그대로 보존했다. 반영본은 outputs/review-audit/reviews.jsonl이다.
- Drive 파일은 직접 변경하지 않았다. 위의 최초 집계는 수정 전 기록이다.
