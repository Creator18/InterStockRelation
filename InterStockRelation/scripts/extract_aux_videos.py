"""
STEP 1 — Extract all 11 relationship columns from pair_cache into a single
compact, Fiedler-aligned video tensor.

Why: relationship_raw.npy (17 GB) and pair_cache/ (16 GB) hold valuable signal
that exists NOWHERE else — multi-timescale return correlations (windows 12/75/
156/375 + EWMA) plus velocity/acceleration. corr_video.npy is ONLY the R_ewma
column. If we delete the big files without extracting these, that signal is
gone forever. This script captures all 11 columns in ~459 MB so the 33 GB of
raw/cache can be safely deleted afterward.

Method (identical to trial2 Cell 11 that built corr_video):
  value[i,j] = column / 100.0  (NaN -> 0), diagonal = 1.0 for the base-correlation
  columns, then reorder rows/cols by the fixed Fiedler order.
  Sampled at the exact snap_timesteps used by corr_video.

Output:
  data/processed/aux_corr_videos.npy   shape (N_SNAPS, 11, 95, 95) float32
  data/processed/aux_corr_meta.pkl     channel names + provenance

Run from project root:  python scripts/extract_aux_videos.py
"""
import os, pickle, time
import numpy as np

ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA  = os.path.join(ROOT, 'data', 'processed')
CACHE = os.path.join(ROOT, 'data', 'pair_cache')

# ── Load metadata ─────────────────────────────────────────────────────────────
with open(os.path.join(DATA, 'pair_meta.pkl'), 'rb') as f:
    pm = pickle.load(f)
with open(os.path.join(DATA, 'video_meta.pkl'), 'rb') as f:
    vm = pickle.load(f)

pairs   = pm['pairs']
tickers = pm['tickers']
cols    = pm['columns']            # 11 column names
tk_idx  = {t: i for i, t in enumerate(tickers)}
fiedler = vm['fiedler_order']
snap_ts = np.asarray(vm['snap_timesteps'])

N        = len(tickers)            # 95
N_PAIRS  = len(pairs)              # 4465
N_SNAPS  = len(snap_ts)            # 1157
N_COLS   = len(cols)              # 11

pair_row = np.array([tk_idx[a] for a, b in pairs], dtype=np.int32)
pair_col = np.array([tk_idx[b] for a, b in pairs], dtype=np.int32)

# Base-correlation columns get diagonal=1; velocity/acceleration columns get
# diagonal=0 (a stock's correlation-velocity with itself is undefined/zero).
BASE_COLS = {'R_single', 'R_short', 'R_medium', 'R_long', 'R_ewma'}

print(f'Stocks   : {N}')
print(f'Pairs    : {N_PAIRS}')
print(f'Snapshots: {N_SNAPS}')
print(f'Columns  : {cols}')
est_mb = N_SNAPS * N_COLS * N * N * 4 / 1e6
print(f'Estimated output size: {est_mb:.0f} MB')

# ── Gather column values at snapshot timesteps from pair_cache ────────────────
# pair_cache/pair_XXXXX.npy has shape (T_bars, 11). We only need the snap rows.
print(f'\nReading {N_PAIRS} pair-cache files (snap rows only)...')
t0 = time.time()
# snap_vals[pair, snap, col]
snap_vals = np.empty((N_PAIRS, N_SNAPS, N_COLS), dtype=np.float32)
for p in range(N_PAIRS):
    if p % 500 == 0 and p > 0:
        el = time.time() - t0
        eta = (N_PAIRS - p) / (p / el)
        print(f'  {p:>5}/{N_PAIRS} | {el:>4.0f}s | ETA {eta:>4.0f}s')
    arr = np.load(os.path.join(CACHE, f'pair_{p:05d}.npy'), mmap_mode='r')
    snap_vals[p] = np.asarray(arr[snap_ts, :])
print(f'  done in {time.time()-t0:.0f}s')

# ── Assemble the (N_SNAPS, N_COLS, N, N) video, Fiedler-reordered ─────────────
print('\nAssembling Fiedler-aligned videos...')
out = np.zeros((N_SNAPS, N_COLS, N, N), dtype=np.float32)
fiedler_ix = np.ix_(fiedler, fiedler)
for ci, cname in enumerate(cols):
    set_diag = 1.0 if cname in BASE_COLS else 0.0
    for s in range(N_SNAPS):
        vals = np.nan_to_num(snap_vals[:, s, ci], nan=0.0) / 100.0
        M = np.zeros((N, N), dtype=np.float32)
        M[pair_row, pair_col] = vals
        M[pair_col, pair_row] = vals
        if set_diag:
            np.fill_diagonal(M, set_diag)
        out[s, ci] = M[fiedler_ix]
    print(f'  [{ci:>2}] {cname:<12} '
          f'min={out[:,ci].min():+.3f} max={out[:,ci].max():+.3f} '
          f'mean={out[:,ci].mean():+.3f}')

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = os.path.join(DATA, 'aux_corr_videos.npy')
np.save(out_path, out)
meta = {
    'channel_names'  : cols,
    'base_cols'      : sorted(BASE_COLS),
    'shape'          : out.shape,
    'fiedler_order'  : fiedler,
    'snap_timesteps' : snap_ts,
    'note'           : 'channel order matches columns; values = column/100, '
                       'Fiedler-reordered, same layout as corr_video.npy. '
                       'R_ewma channel == corr_video.npy (verification).',
}
with open(os.path.join(DATA, 'aux_corr_meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

actual_mb = os.path.getsize(out_path) / 1e6
print(f'\nSaved {out_path}')
print(f'  shape {out.shape}  size {actual_mb:.0f} MB')

# ── Verification: R_ewma channel should equal corr_video.npy exactly ──────────
cv = np.load(os.path.join(DATA, 'corr_video.npy'), mmap_mode='r')
ewma_ch = cols.index('R_ewma')
diff = float(np.max(np.abs(out[:, ewma_ch] - np.asarray(cv))))
print(f'\nVERIFICATION: extracted R_ewma vs corr_video.npy  max_abs_diff = {diff:.2e}')
print('  ' + ('EXACT MATCH - extraction pipeline confirmed correct.'
              if diff < 1e-5 else
              'MISMATCH - investigate before deleting anything!'))
