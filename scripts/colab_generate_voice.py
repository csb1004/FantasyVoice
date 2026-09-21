"""10 cell 2: rerun this cell for new dialogue without reloading models."""
TEXT = '안녕하세요. 오늘도 함께해 주셔서 감사합니다.'
CHARACTER = 'Adela_KOR'  # Use a name printed by the loading cell.
SEED = 42
DOWNLOAD = True

if 'generate_voice' not in globals():
    raise RuntimeError('10번 노트북의 첫 번째 모델 로딩 셀을 먼저 실행하세요.')
result = generate_voice(TEXT, CHARACTER, seed=SEED, download=DOWNLOAD)
