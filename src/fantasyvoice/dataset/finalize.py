"""CPU-only result validation/export. Never invokes ASR or touches source WAVs."""
from collections import Counter
import json
import os
from pathlib import Path
import tempfile
import zipfile

from .full_dataset import eligible_rows, export_dataset, _digest, _identity, _load_shards, _latest
from .storage import read_jsonl, write_json, write_jsonl, atomic_text


def finalize_dataset(rows, output, archive_path, reviews=()):
    output, archive_path = Path(output), Path(archive_path)
    config = json.loads((output / 'run.json').read_text(encoding='utf-8'))
    dataset = json.loads((output / 'dataset.json').read_text(encoding='utf-8'))
    sources = eligible_rows(rows)
    signature = _digest([{k: r[k] for k in ('audio_path', 'character_id', 'sha256', 'duration_seconds')} for r in sources])
    if dataset.get('inventory_signature') != signature or dataset.get('count') != len(sources):
        raise ValueError('Reference inventory differs from the transcription run')
    expected = {_identity(config, r): r for r in sources}
    if len(expected) != len(sources):
        raise ValueError('Duplicate source identities in inventory')
    shards = _load_shards(output)
    for records in shards.values():
        for result in records:
            source = expected.get(result.get('key'))
            if source is None or any(result.get(k) != source[k] for k in ('audio_path', 'character_id', 'sha256')):
                raise ValueError('Checkpoint identity does not match run config and inventory')
            if result.get('status') not in {'success', 'needs_review', 'error'}:
                raise ValueError('Unknown checkpoint status')
            if result['status'] == 'success' and (not isinstance(result.get('asr', {}).get('text'), str)
                                                or not isinstance(result['asr'].get('segments'), list)):
                raise ValueError('Invalid successful ASR checkpoint')
    latest = _latest(shards)
    missing = [dict(r, retry_reason='not_processed') for k, r in expected.items() if k not in latest]
    failures = [dict(expected[k], retry_reason=r.get('error', 'asr_error'))
                for k, r in latest.items() if r['status'] == 'error']
    merged_reviews = read_jsonl(output / 'imported-reviews.jsonl') + read_jsonl(output / 'reviews.jsonl') + list(reviews)
    for review in merged_reviews:
        if review.get('decision') not in {'accepted', 'rejected', 'pending'} or not isinstance(review.get('corrected_text'), str):
            raise ValueError('Invalid review record')
    # No input changes are made until all checkpoint identities have been checked.
    export = export_dataset(rows, output, merged_reviews)
    write_jsonl(output / 'reviews-applied.jsonl', merged_reviews)
    write_jsonl(output / 'retry.jsonl', missing + failures)
    counts = Counter(r['status'] for r in latest.values())
    overruns = []
    for key, r in latest.items():
        end = max((s.get('end', 0) for s in r.get('asr', {}).get('segments', [])), default=0)
        duration = expected[key]['duration_seconds']
        if end > duration:
            overruns.append({'audio_path': r['audio_path'], 'duration_seconds': duration, 'segment_end_seconds': end})
    write_jsonl(output / 'segment-overruns.jsonl', overruns)
    characters = {}
    for key, source in expected.items():
        entry = characters.setdefault(source['character_id'], Counter())
        entry['total'] += 1
        entry[latest[key]['status'] if key in latest else 'unprocessed'] += 1
    report = {**export, 'total': len(sources), 'success': counts['success'],
              'needs_review': counts['needs_review'], 'error': counts['error'],
              'unprocessed': len(missing), 'asr_pass_complete': not missing and not failures,
              'segment_endpoint_overruns': len(overruns), 'per_character': characters,
              'scope': 'checkpoint_integrity_only_no_new_audio_or_transcription_validation'}
    write_json(output / 'completion-report.json', report)
    lines = ['# 전체 전사 결과 점검', '',
             f"- 대상: {len(sources):,}개", f"- 전사 성공: {counts['success']:,}개",
             f"- 무음·빈 전사 등 검수 대기: {counts['needs_review']:,}개",
             f"- 처리 오류: {counts['error']:,}개 / 미처리: {len(missing):,}개",
             f"- 사람 대사 확인: {export['human_accepted_count']:,}개",
             f"- ASR 구간 끝이 원본 길이를 넘는 파일: {len(overruns):,}개",
             '', '전사 성공은 대사 정확도 보증이 아닙니다. 감정·운율 라벨, 데이터 분할,',
             '자동 전사 품질 기준은 아직 확정되지 않아 최종 학습 완료 데이터셋이 아닙니다.',
             '원본 음성은 이 단계에서 다시 읽거나 수정하지 않았습니다.', '',
             '오류/미처리가 남았다면 기존 전체 전사 코드를 같은 출력 폴더로 다시 실행하세요.',
             '검수 대기 항목은 단순 재시도로 자동 합격 처리되지 않습니다.', '']
    atomic_text(output / 'completion-report.md', '\n'.join(lines))
    # Strict allowlist: no audio, caches, unrelated files or previous archives.
    names = ['run.json', 'dataset.json', 'progress.json', 'imported-reviews.jsonl', 'reviews.jsonl',
             'reviews-applied.jsonl', 'excluded-duration.jsonl', 'candidates.jsonl', 'reviewed.jsonl',
             'needs-review.jsonl', 'retry.jsonl', 'export-summary.json', 'segment-overruns.jsonl',
             'completion-report.json', 'completion-report.md']
    files = [output / name for name in names if (output / name).is_file()]
    files += sorted((output / 'results').glob('*.jsonl'))
    files += sorted(output.glob('runtime-*.json'))
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=archive_path.parent, suffix='.zip', delete=False) as stream:
        temp = Path(stream.name)
    try:
        with zipfile.ZipFile(temp, 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in files:
                archive.write(path, 'dataset-v1/' + path.relative_to(output).as_posix())
        os.replace(temp, archive_path)
    finally:
        temp.unlink(missing_ok=True)
    return report
