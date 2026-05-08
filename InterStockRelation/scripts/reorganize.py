"""
reorganize.py
Moves all Trial 1 outputs into a subfolder: trial1_archive/
Keeps only the files needed for Trial 2 in the root project directory.

Files KEPT in root (reused in Trial 2):
    - archive/                  raw stock CSVs
    - pair_cache/               4465 individual pair .npy files
    - relationship_raw.npy      full mmap array
    - pair_meta.pkl             pairs, tickers, timestamps, column names

Everything else moves to trial1_archive/

Run from the project directory:
    python reorganize.py
"""

import os
import shutil

PROJECT_DIR  = r"C:\Users\HP\Documents\InterStockRelation_Paper\InterStockRelation"
ARCHIVE_DIR  = os.path.join(PROJECT_DIR, "trial1_archive")

# ── Items to KEEP in root ─────────────────────────────────────────────────────
KEEP = {
    "archive",
    "pair_cache",
    "relationship_raw.npy",
    "pair_meta.pkl",
    "trial1_archive",       # don't move the archive into itself
    "reorganize.py",        # don't move this script
    "trial1.ipynb",         # keep the notebook in root
}

# ── Create archive folder ─────────────────────────────────────────────────────
os.makedirs(ARCHIVE_DIR, exist_ok=True)
print(f"Archive folder : {ARCHIVE_DIR}")
print(f"Project dir    : {PROJECT_DIR}\n")

# ── Scan root and move everything not in KEEP ─────────────────────────────────
moved   = []
skipped = []
errors  = []

entries = os.listdir(PROJECT_DIR)
for name in sorted(entries):
    if name in KEEP:
        skipped.append(name)
        continue

    src = os.path.join(PROJECT_DIR, name)
    dst = os.path.join(ARCHIVE_DIR, name)

    try:
        shutil.move(src, dst)
        moved.append(name)
        kind = "DIR " if os.path.isdir(dst) else "FILE"
        print(f"  MOVED  [{kind}]  {name}")
    except Exception as e:
        errors.append((name, str(e)))
        print(f"  ERROR         {name}  →  {e}")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n── Summary ──")
print(f"  Moved   : {len(moved)} items")
print(f"  Kept    : {len(skipped)} items  {skipped}")
print(f"  Errors  : {len(errors)}")

if errors:
    print(f"\n── Errors ──")
    for name, err in errors:
        print(f"  {name} : {err}")

print(f"\n── Kept in root ──")
for name in sorted(os.listdir(PROJECT_DIR)):
    kind = "DIR " if os.path.isdir(os.path.join(PROJECT_DIR, name)) else "FILE"
    print(f"  [{kind}]  {name}")

print(f"\n── Moved to trial1_archive/ ──")
for name in sorted(os.listdir(ARCHIVE_DIR)):
    kind = "DIR " if os.path.isdir(os.path.join(ARCHIVE_DIR, name)) else "FILE"
    print(f"  [{kind}]  {name}")

print(f"\n✓ Reorganization complete.")