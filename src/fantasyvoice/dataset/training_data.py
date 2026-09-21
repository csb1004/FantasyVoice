"""Explicitly configured manifest preparation; no audio conversion or model training."""
import math
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, pstdev

from .storage import write_json, write_jsonl
from fantasyvoice.style.runner import digest


def check_settings(config):
    speed_limit = config.get('max_speed_phonemes_per_second')
    if speed_limit is not None and (type(speed_limit) not in (int, float)
            or not math.isfinite(speed_limit) or speed_limit <= 0):
        raise ValueError('Invalid speed limit')
    required = ('validation_fraction', 'seed', 'allow_pending_transcripts', 'label_policy',
                'pitch_baseline', 'grouping', 'no_pitch_policy')
    if any(config.get(k) is None for k in required):
        raise ValueError('Confirm all dataset settings before running 06')
    if not 0 < config['validation_fraction'] < 1 or type(config['seed']) is not int:
        raise ValueError('Invalid split settings')
    if type(config['allow_pending_transcripts']) is not bool:
        raise ValueError('allow_pending_transcripts must be an explicit boolean')
    if config['label_policy'] not in ('complete_only', 'masked_partial'):
        raise ValueError('Unknown label_policy')
    if (config['pitch_baseline'] != 'median' or config['no_pitch_policy'] != 'hold_out'
            or config['grouping'] != 'character_text_and_audio_hash'):
        raise ValueError('Unsupported settings; do not substitute a different policy silently')


def grouped_split(rows, fraction, seed):
    """Join transitive text/hash duplicates, then greedily balance character counts."""
    parents = list(range(len(rows)))
    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    seen = {}
    for i, row in enumerate(rows):
        text = ' '.join(unicodedata.normalize('NFC', row['text']).split())
        for key in (('text', row['character_id'], text), ('audio', row['sha256'])):
            if key in seen:
                parents[root(i)] = root(seen[key])
            seen[key] = i
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[root(i)].append(row)
    ordered = sorted(groups.values(), key=lambda group: digest({
        'seed': seed, 'paths': sorted(r['audio_path'] for r in group)}))
    total = Counter(r['character_id'] for r in rows)
    remaining = total.copy()
    validation = Counter()
    assigned = {}
    for group in ordered:
        counts = Counter(r['character_id'] for r in group)
        # Keep at least one training observation per character. Ratios are approximate.
        improvement = sum(abs(validation[c] - fraction * total[c]) -
                          abs(validation[c] + count - fraction * total[c]) for c, count in counts.items())
        use_validation = improvement > 0 and all(remaining[c] > count for c, count in counts.items())
        if use_validation:
            validation.update(counts)
            remaining.subtract(counts)
        group_id = digest(sorted((r['audio_path'], r['sha256']) for r in group))
        for row in group:
            assigned[row['audio_path']] = ('validation' if use_validation else 'train', group_id)
    return assigned


def prepare_training_data(rows, config, output):
    check_settings(config)
    rows = list(rows)
    if not rows or len({r['audio_path'] for r in rows}) != len(rows):
        raise ValueError('Expected nonempty candidates with unique audio paths')
    if any(r.get('split') is not None for r in rows):
        raise ValueError('Input already has split assignments; refusing to repartition')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use an empty output folder to preserve previous dataset versions')
    eligible, held = [], []
    for row in sorted(rows, key=lambda r: r['audio_path']):
        reason = None
        state = row.get('review_status')
        if state != 'accepted' and not (state == 'pending' and config['allow_pending_transcripts']):
            reason = 'transcript_review_policy'
        elif row.get('validation_errors') != [] or row.get('style_status') not in ('generated', 'partial'):
            reason = 'style_not_validated'
        else:
            masks = row['pseudo_style']['valid_targets']
            if config['label_policy'] == 'complete_only' and not all(masks.get(k) for k in ('emotion', 'speed', 'pitch_raw', 'pause')):
                reason = 'incomplete_style_targets'
            speed = row['pseudo_style']['speed_phonemes_per_second']
            limit = config.get('max_speed_phonemes_per_second')
            if reason is None and limit is not None and speed is not None and speed > limit:
                reason = 'speed_above_limit'
        if reason:
            held.append(dict(row, hold_reason=reason))
        else:
            eligible.append(row)
    if not eligible:
        raise ValueError('No eligible rows under the selected policies')
    assigned = grouped_split(eligible, config['validation_fraction'], config['seed'])
    train_pitch = defaultdict(list)
    for row in eligible:
        if assigned[row['audio_path']][0] == 'train':
            pitch = row['pseudo_style']['pitch_f0_median_hz']
            if pitch is not None:
                if not math.isfinite(pitch) or pitch <= 0:
                    raise ValueError('Invalid training pitch')
                train_pitch[row['character_id']].append(pitch)
    baselines = {c: median(values) for c, values in sorted(train_pitch.items())}
    prepared, train_values = [], defaultdict(list)
    for row in eligible:
        character = row['character_id']
        if character not in baselines:
            held.append(dict(row, hold_reason='no_training_pitch_baseline'))
            continue
        split, group_id = assigned[row['audio_path']]
        value = row['pseudo_style']
        pitch = value['pitch_f0_median_hz']
        raw = {'speed': value['speed_phonemes_per_second'],
               'pitch': 12 * math.log2(pitch / baselines[character]) if pitch is not None else None,
               'pause': value['pause_ratio']}
        if split == 'train':
            for key, number in raw.items():
                if number is not None:
                    if not math.isfinite(number):
                        raise ValueError('Non-finite style target')
                    train_values[key].append(number)
        prepared.append((row, split, group_id, raw))
    normalization = {}
    for key in ('speed', 'pitch', 'pause'):
        values = train_values[key]
        std = pstdev(values) if values else None
        normalization[key] = {'count': len(values), 'mean': mean(values) if values else None,
                              'std': std, 'scale': (std if std else 1.) if values else None,
                              'zero_variance': std == 0 if values else False}
    manifests = {'train': [], 'validation': []}
    for row, split, group_id, raw in prepared:
        style = {'emotion_probabilities': row['pseudo_style']['emotion']['probabilities'],
                 'pitch_semitones': raw['pitch'], 'valid_targets': {}}
        style['valid_targets']['emotion'] = style['emotion_probabilities'] is not None
        for key, number in raw.items():
            stat = normalization[key]
            valid = number is not None and stat['count'] > 0
            style[key + '_z'] = (number - stat['mean']) / stat['scale'] if valid else None
            style['valid_targets'][key] = valid
        manifests[split].append(dict(row, split=split, split_group=group_id, training_style=style))
    if not manifests['train'] or not manifests['validation']:
        raise ValueError('Selected policies/group sizes leave an empty split; review settings/data')
    characters = sorted({r['character_id'] for r in manifests['train']})
    char_map = {name: i for i, name in enumerate(characters)}
    per_character = {name: {split: sum(r['character_id'] == name for r in data)
                            for split, data in manifests.items()} for name in characters}
    # Verify both kinds of group leakage before exporting anything.
    for get_key in (lambda r: r['sha256'], lambda r: (r['character_id'], ' '.join(unicodedata.normalize('NFC', r['text']).split()))):
        if {get_key(r) for r in manifests['train']} & {get_key(r) for r in manifests['validation']}:
            raise ValueError('Split leakage detected')
    report = {'input_count': len(rows), 'train_count': len(manifests['train']),
              'validation_count': len(manifests['validation']), 'held_count': len(held),
              'actual_validation_fraction': len(manifests['validation']) / len(prepared),
              'hold_reasons': dict(Counter(r['hold_reason'] for r in held)),
              'per_character': per_character,
              'characters_without_validation': [c for c, counts in per_character.items() if counts['validation'] == 0],
              'pitch_baselines_hz': baselines, 'normalization': normalization,
              'statistics_fitted_on': 'train_only', 'config': config,
              'source_signature': digest(rows), 'trainer_implemented': False,
              'audio_preprocessing_applied': False}
    for split, data in manifests.items():
        write_jsonl(output / f'{split}.jsonl', data)
    write_jsonl(output / 'held-out.jsonl', held)
    write_json(output / 'character-map.json', char_map)
    write_json(output / 'dataset-preparation.json', report)
    write_json(output / 'config.json', config)
    return report
