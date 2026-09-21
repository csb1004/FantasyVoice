"""CPU-only style checkpoint audit; never fit statistics or approve training rows."""
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
import zipfile

from fantasyvoice.dataset.storage import read_jsonl, write_json, write_jsonl, atomic_text
from .features import emotion_probabilities, summarize
from .runner import digest, label_key


def validate_style(value):
    """Check structural invariants, not subjective quality thresholds."""
    errors = []
    try:
        duration = value['decoded_duration_seconds']
        expected = summarize(value['speech_intervals_seconds'], duration, value['phoneme_count'], [])
        for name in ('speech_seconds', 'speed_phonemes_per_second', 'pause_ratio'):
            actual, wanted = value[name], expected[name]
            if (actual is None) != (wanted is None) or (wanted is not None and
                    (not isinstance(actual, (int, float)) or not math.isfinite(actual)
                     or not math.isclose(actual, wanted, rel_tol=1e-5, abs_tol=1e-6))):
                errors.append(f'{name}: inconsistent with speech intervals/phoneme count')
        pitch = value['pitch_f0_median_hz']
        if pitch is not None and (not isinstance(pitch, (int, float)) or not math.isfinite(pitch) or pitch <= 0):
            errors.append('pitch_f0_median_hz: must be positive and finite')
        if value['pitch_semitones'] is not None:
            errors.append('pitch_semitones: expected null before training split')
        emotion = value['emotion']
        raw = emotion['raw_probabilities']
        wanted = emotion_probabilities(list(raw), list(raw.values())) if raw else {
            'raw_probabilities': None, 'probabilities': None, 'top_label': None}
        probabilities, expected_probabilities = emotion['probabilities'], wanted['probabilities']
        matches = probabilities is None and expected_probabilities is None
        if isinstance(probabilities, dict) and isinstance(expected_probabilities, dict):
            matches = probabilities.keys() == expected_probabilities.keys() and all(
                isinstance(probabilities[k], (int, float)) and math.isfinite(probabilities[k])
                and 0 <= probabilities[k] <= 1
                and math.isclose(probabilities[k], expected_probabilities[k], rel_tol=1e-7, abs_tol=1e-12)
                for k in expected_probabilities)
        if emotion['top_label'] != wanted['top_label'] or not matches:
            errors.append('emotion: invalid normalization/top label')
        masks = {'emotion': emotion['probabilities'] is not None,
                 'speed': value['speed_phonemes_per_second'] is not None,
                 'pitch_raw': pitch is not None, 'pitch_normalized': False,
                 'pause': value['pause_ratio'] is not None}
        if value['valid_targets'] != masks:
            errors.append('valid_targets: mismatch with available values')
        if not isinstance(value['issues'], list):
            errors.append('issues: expected a list')
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        errors.append(f'malformed_style: {exc}')
    return errors


def distribution(values):
    values = sorted(values)
    if not values:
        return {'count': 0}
    def quantile(q):
        position = (len(values) - 1) * q
        low, high = math.floor(position), math.ceil(position)
        return values[low] + (values[high] - values[low]) * (position - low)
    return {'count': len(values), 'min': values[0], 'p05': quantile(.05),
            'median': median(values), 'p95': quantile(.95), 'max': values[-1]}


def finalize_styles(rows, source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Report output must be separate from source checkpoints')
    rows = list(rows)
    if not rows or len({r['audio_path'] for r in rows}) != len(rows):
        raise ValueError('Candidates must be nonempty and have unique paths')
    config = json.loads((source / 'config.json').read_text(encoding='utf-8'))
    cache = {}
    checkpoint_files = sorted((source / 'results').glob('*.jsonl'))
    for path in checkpoint_files:
        for record in read_jsonl(path):
            key = record['style_key']
            if key in cache:
                raise ValueError(f'Duplicate checkpoint key: {key}')
            cache[key] = record
    data, retry, review = [], [], []
    counts, issue_counts, emotion_counts, available = Counter(), Counter(), Counter(), Counter()
    per_character, numeric = defaultdict(Counter), defaultdict(list)
    expected_keys, invalid = set(), 0
    for row in rows:
        key = label_key(row, config)
        expected_keys.add(key)
        record = cache.get(key, {'style_key': key, 'style_status': 'pending'})
        # Ignore checkpoint fields outside the runner's schema; candidates own text/review/split.
        result = dict(row, **{k: v for k, v in record.items() if k in (
            'style_key', 'style_status', 'pseudo_style', 'style_error', 'style_reason')})
        status = result['style_status']
        errors = []
        if status in ('generated', 'partial'):
            value = result.get('pseudo_style', {})
            errors = validate_style(value)
            if not errors:
                if (status == 'partial') != bool(value['issues']):
                    errors.append('style_status: inconsistent with issues')
        elif status not in ('pending', 'error', 'excluded'):
            errors.append(f'Unknown style_status: {status}')
        result['validation_errors'] = errors
        invalid += bool(errors)
        counts[status] += 1
        per_character[row['character_id']][status] += 1
        if errors or status in ('error', 'pending'):
            retry.append(result)
        if errors or status in ('error', 'pending', 'partial', 'excluded'):
            review.append(result)
        if status in ('generated', 'partial') and not errors:
            value = result['pseudo_style']
            issue_counts.update(value['issues'])
            emotion_counts[value['emotion']['top_label'] or 'unavailable'] += 1
            available.update(k for k, valid in value['valid_targets'].items() if valid)
            for name in ('speed_phonemes_per_second', 'pitch_f0_median_hz', 'pause_ratio', 'speech_seconds'):
                number = value[name]
                if number is not None:
                    numeric[name].append(number)
        data.append(result)
    report = {
        'total': len(rows), 'counts': dict(counts), 'invalid_count': invalid,
        'style_pass_complete': not retry, 'training_ready': False,
        'candidate_review_states': dict(Counter(r['review_status'] for r in rows)),
        'issues': dict(issue_counts), 'emotion_top_labels': dict(emotion_counts),
        'valid_target_counts': dict(available),
        'numeric': {key: distribution(values) for key, values in numeric.items()},
        'per_character': dict(per_character),
        'source_signature': digest(rows), 'config_sha256': digest(config),
        'unmatched_checkpoint_count': len(set(cache) - expected_keys),
        'remaining': ['ASR quality acceptance policy', 'train/validation split',
                      'train-only character pitch baselines and standardization', 'training configuration'],
        'scope': 'Checkpoint and numeric consistency only; no audio listening or model rerun',
    }
    previous_path = source / 'summary.json'
    if previous_path.exists():
        previous = json.loads(previous_path.read_text(encoding='utf-8'))
        report['previous_summary_matches'] = all(previous.get(key) == report[key] for key in
            ('source_signature', 'config_sha256', 'counts', 'total'))
    write_json(output / 'completion-report.json', report)
    write_json(output / 'config.json', config)
    write_jsonl(output / 'style-labels.jsonl', data)
    write_jsonl(output / 'retry.jsonl', retry)
    write_jsonl(output / 'style-needs-review.jsonl', review)
    lines = ['# 스타일 라벨 결과 점검', '', f"대상: {len(rows):,}개",
             f"상태별 개수: {dict(counts)}", f"형식/계산 이상: {invalid:,}개",
             f"재시도/확인 대상: {len(retry):,}개", f"분석 처리 완료: {not retry}", '',
             'partial은 일부 목표값 누락으로 별도 확인이 필요합니다. 자동 제외하지 않았습니다.',
             '검수 상태를 유지했습니다. 분석 완료는 최종 학습 데이터 승인이나 학습 완료가 아닙니다.',
             '피치 정규화와 학습 데이터 분할은 아직 적용하지 않았습니다.', '',
             '## 감정 최상위 라벨 분포', '', json.dumps(dict(emotion_counts), ensure_ascii=False), '',
             '## 수치 분포 (품질 합격 기준 아님)', '',
             json.dumps(report['numeric'], ensure_ascii=False, indent=2), '']
    atomic_text(output / 'completion-report.md', '\n'.join(lines))
    # Metadata only; originals retained to diagnose discrepancies or failed records.
    archive = output / 'style-report.zip'
    temp_archive = output / 'style-report.zip.partial'
    names = ('completion-report.json', 'completion-report.md', 'config.json',
             'style-labels.jsonl', 'retry.jsonl', 'style-needs-review.jsonl')
    with zipfile.ZipFile(temp_archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for name in names:
            z.write(output / name, 'style-v1/' + name)
        for path in checkpoint_files:
            z.write(path, 'style-v1/results/' + path.name)
        if previous_path.exists():
            z.write(previous_path, 'style-v1/original-summary.json')
    temp_archive.replace(archive)
    return report
