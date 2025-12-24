import csv
from pathlib import Path

csv_path = Path('MedCLIP-SAMv2-finetune/data/medpix_dataset/medpix_dataset.csv')
if not csv_path.exists():
    print('CSV not found:', csv_path)
    raise SystemExit(1)

ds_root = csv_path.parent
try:
    repo_root = csv_path.parents[2]
except Exception:
    repo_root = ds_root.parent

images_dir = ds_root / 'images'

matched = 0
missing = 0
matched_examples = []
missing_examples = []

with csv_path.open('r', newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        raw_path = row['filename'].replace('\\', '/').replace('data/data/medpix_dataset', 'data/medpix_dataset')
        p = Path(raw_path)
        candidates = []
        if p.is_absolute():
            candidates.append(p)
        else:
            if 'data/medpix_dataset' in raw_path:
                candidates.append(repo_root / raw_path)
            candidates.append(ds_root / raw_path)
            candidates.append(ds_root / 'images' / Path(raw_path).name)
        chosen = None
        for c in candidates:
            if c.exists():
                chosen = c
                break
        if chosen is None:
            # basename lookup
            candidate2 = images_dir / Path(raw_path).name
            if candidate2.exists():
                chosen = candidate2
        if chosen is None:
            # recursive search (stop after first found)
            try:
                found = next(images_dir.rglob(Path(raw_path).name))
                chosen = found
            except StopIteration:
                chosen = None
        if chosen is None:
            missing += 1
            if len(missing_examples) < 20:
                missing_examples.append(raw_path)
        else:
            matched += 1
            if len(matched_examples) < 20:
                matched_examples.append(str(chosen))

report = Path('logs/medpix_path_check.txt')
report.parent.mkdir(exist_ok=True)
with report.open('w', encoding='utf-8') as out:
    out.write(f'Matched: {matched}\n')
    out.write(f'Missing: {missing}\n')
    out.write('\nMatched examples:\n')
    for m in matched_examples:
        out.write(m + '\n')
    out.write('\nMissing examples (raw CSV paths):\n')
    for m in missing_examples:
        out.write(m + '\n')

print(f'Report written: {report} (Matched: {matched}, Missing: {missing})')
