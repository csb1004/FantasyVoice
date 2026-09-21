import argparse
import json
from pathlib import Path

from .dataset.inventory import scan, select_pilot, summarize
from .dataset.storage import check_output, read_jsonl, write_json, write_jsonl


def main():
    parser = argparse.ArgumentParser(description='FantasyVoice inventory / Korean ASR pilot')
    commands = parser.add_subparsers(dest='command', required=True)
    inventory = commands.add_parser('inventory')
    inventory.add_argument('--root', type=Path, required=True)
    inventory.add_argument('--output', type=Path, required=True)
    inventory.add_argument('--per-character', type=int, default=3)
    asr = commands.add_parser('transcribe')
    asr.add_argument('--root', type=Path, required=True)
    asr.add_argument('--selection', type=Path, required=True)
    asr.add_argument('--output', type=Path, required=True)
    asr.add_argument('--config', type=Path, required=True)
    asr.add_argument('--model-cache', type=Path, required=True)
    args = parser.parse_args()
    check_output(args.root, args.output)
    if args.command == 'inventory':
        if args.per_character < 1:
            parser.error('--per-character must be positive')
        rows = scan(args.root, lambda n, total: print(f'Inventory {n}/{total}', flush=True))
        write_jsonl(args.output / 'inventory.jsonl', rows)
        write_json(args.output / 'summary.json', summarize(rows))
        selection = select_pilot(rows, args.per_character)
        write_jsonl(args.output / 'pilot.jsonl', selection)
        print(f'{len(rows)} files; {len(selection)} pilot samples. Headers only; no ASR run.')
    else:
        from .dataset.transcribe import WhisperBackend, run_pilot
        if not args.selection.is_file():
            parser.error('Selection file does not exist')
        rows = read_jsonl(args.selection)
        if not rows:
            parser.error('Selection is empty')
        check_output(args.root, args.model_cache)
        options = json.loads(args.config.read_text(encoding='utf-8'))['whisper_options']
        backend = WhisperBackend(options, args.model_cache)
        results = run_pilot(args.root, rows, args.output, backend.config, backend,
                            lambda n, total, state: print(f'ASR {n}/{total}: {state}', flush=True))
        latest = {r['key']: r for r in results}
        failures = sum(r['status'] == 'error' for r in latest.values())
        print(f'{len(latest)} result identities; {failures} failures. Listening review required.')
        if failures:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
