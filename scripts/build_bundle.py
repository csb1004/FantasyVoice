"""Build code + local reference inventory; never include source audio or transcripts."""
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parents[1]
target = root / 'dist' / 'fantasyvoice-pilot.zip'
target.parent.mkdir(exist_ok=True)
files = [root / 'pyproject.toml', root / 'README.md']
reference = root / 'outputs/inventory/inventory.jsonl'
if not reference.is_file():
    raise SystemExit('Generate the local inventory before building the Colab bundle')
for folder in ['src', 'config', 'notebooks', 'tests', 'docs', 'scripts']:
    files.extend(p for p in (root / folder).rglob('*')
                 if p.is_file() and '__pycache__' not in p.parts
                 and not any(part.endswith('.egg-info') for part in p.parts))
with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(files):
        archive.write(path, path.relative_to(root).as_posix())
    archive.write(reference, 'reference/inventory.jsonl')
    reviews = root / 'outputs/review-audit/reviews-linked.jsonl'
    if reviews.is_file():
        archive.write(reviews, 'reference/reviews.jsonl')
print(target)
