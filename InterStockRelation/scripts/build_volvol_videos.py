"""
STEP 2 — Build VOLUME-correlation and VOLATILITY-correlation videos from the
raw 5-min CSVs, perfectly aligned with corr_video.npy.

These signals do NOT exist anywhere in the processed data (relationship_raw is
returns-only). The volume_signal_check showed volume/volatility correlations are
largely independent of return correlation (avg cross-corr ~0.18 / ~0.25), so
they are genuinely new information for Trial 7.

Method — identical pipeline to corr_video, only the input series changes:
  * Reindex each stock's OHLCV onto the exact processed timeline (pair_meta
    timestamps, 87122 bars — verified 0% NaN coverage).
  * Per-stock signals:
      volume signal     = d log(volume)        (overnight increments masked to 0)
      volatility signal = log(high-low range / close + eps)
  * EWMA correlation per pair, alpha = 1-0.94 = 0.06  (SAME as R_ewma / corr_video).
    ewma_corr internally demeans with the time-varying EWMA mean, which removes
    slow trends / intraday level drift.
  * Sample at the exact snap_timesteps, set diagonal = 1, reorder by the SAME
    Fiedler order.  Output is frame-for-frame aligned with corr_video.

Outputs (~42 MB each):
  data/processed/volume_corr_video.npy   (N_SNAPS, 95, 95) float32
  data/processed/volat_corr_video.npy    (N_SNAPS, 95, 95) float32

NOTE: signal definitions above are deliberate but reviewable — if the volume
co-movement looks seasonality-inflated in the output, switch volatility to a
differenced form too. Inspect before committing to Trial 7.

Run from project root:  python scripts/build_volvol_videos.py
"""
import os, pickle, time
import numpy as np
import pandas as pd
from scipy.signal import lfilter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA = os.path.join(ROOT, 'data', 'processed')
RAW  = os.path.join(ROOT, 'data', 'raw')

EWMA_ALPHA = 1 - 0.94   # 0.06, identical to the return-correlation pipeline

# ── Metadata ──────────────────────────────────────────────────────────────────
with open(os.path.join(DATA, 'pair_meta.pkl'), 'rb') as f:
    pm = pickle.load(f)
with open(os.path.join(DATA, 'video_meta.pkl'), 'rb') as f:
    vm = pickle.load(f)

pairs    = pm['pairs']
tickers  = pm['tickers']
ts       = pm['timestamps']            # processed timeline (87122 bars)
tk_idx   = {t: i for i, t in enumerate(tickers)}
fiedler  = vm['fiedler_order']
snap_ts  = np.asarray(vm['snap_timesteps'])

N, N_PAIRS, N_SNAPS = len(tickers), len(pairs), len(snap_ts)
T = len(ts)
pair_i = np.array([tk_idx[a] for a, b in pairs], dtype=np.int32)
pair_j = np.array([tk_idx[b] for a, b in pairs], dtype=np.int32)

print(f'Stocks {N} | pairs {N_PAIRS} | bars {T} | snaps {N_SNAPS} | alpha {EWMA_ALPHA:.3f}')

# ── Load raw OHLCV, reindex onto processed timeline ──────────────────────────
print('\nLoading raw CSVs and building signal matrices...')
log_vol  = np.full((T, N), np.nan, dtype=np.float64)
log_rng  = np.full((T, N), np.nan, dtype=np.float64)
day_id   = pd.Series(ts).dt.normalize().values     # for overnight masking

t0 = time.time()
for tk, k in tk_idx.items():
    fn = os.path.join(RAW, f'{tk}_5minute.csv')
    d = pd.read_csv(fn, parse_dates=['date']).set_index('date').reindex(ts)
    v = d['volume'].to_numpy(dtype=np.float64)
    log_vol[:, k] = np.log(np.clip(v, 1.0, None))
    rng = (d['high'].to_numpy() - d['low'].to_numpy()) / d['close'].to_numpy()
    log_rng[:, k] = np.log(np.clip(rng, 1e-6, None))
print(f'  loaded {N} stocks in {time.time()-t0:.0f}s')

# Volume signal = d log(volume), overnight masked; Volatility signal = log range
dvol = np.diff(log_vol, axis=0, prepend=log_vol[:1])
overnight = np.concatenate([[True], day_id[1:] != day_id[:-1]])
dvol[overnight] = 0.0
vol_sig = np.nan_to_num(dvol,    nan=0.0)
rng_sig = np.nan_to_num(log_rng, nan=0.0)

# ── Vectorized EWMA correlation (matches pandas ewm adjust=False) ─────────────
def ewma_axis0(X, alpha):
    """EWMA along time axis (axis 0), adjust=False, y[0]=X[0]. X: (T, M)."""
    b = np.array([alpha], dtype=np.float64)
    a = np.array([1.0, -(1.0 - alpha)], dtype=np.float64)
    zi = ((1.0 - alpha) * X[0])[None, :]      # state so y[0] == X[0]
    y, _ = lfilter(b, a, X, axis=0, zi=zi)
    return y

def build_corr_video(sig, name):
    print(f'\n[{name}] computing EWMA correlation...')
    t0 = time.time()
    m   = ewma_axis0(sig, EWMA_ALPHA)             # (T, N) time-varying means
    dx  = sig - m
    var = ewma_axis0(dx * dx, EWMA_ALPHA)         # (T, N)
    var = np.clip(var, 1e-12, None)

    # Cov per pair = EWMA(dx_i * dx_j); process in chunks to bound memory
    video = np.zeros((N_SNAPS, N, N), dtype=np.float32)
    CHUNK = 600
    for start in range(0, N_PAIRS, CHUNK):
        end = min(start + CHUNK, N_PAIRS)
        pi, pj = pair_i[start:end], pair_j[start:end]
        prod = dx[:, pi] * dx[:, pj]              # (T, chunk)
        cov  = ewma_axis0(prod, EWMA_ALPHA)       # (T, chunk)
        denom = np.sqrt(var[:, pi] * var[:, pj])
        corr = (cov / denom)[snap_ts]             # (N_SNAPS, chunk)
        corr = np.clip(np.nan_to_num(corr, nan=0.0), -1.0, 1.0)
        for c, p in enumerate(range(start, end)):
            video[:, pi[c], pj[c]] = corr[:, c]
            video[:, pj[c], pi[c]] = corr[:, c]
        if start % (CHUNK * 3) == 0:
            print(f'  pairs {end:>5}/{N_PAIRS} | {time.time()-t0:.0f}s')

    # diagonal = 1, then Fiedler reorder (same layout as corr_video)
    for s in range(N_SNAPS):
        np.fill_diagonal(video[s], 1.0)
    video = video[:, fiedler][:, :, fiedler]
    print(f'  done in {time.time()-t0:.0f}s | '
          f'min={video.min():+.3f} max={video.max():+.3f} mean={video.mean():+.3f} '
          f'frac_neg={(video[:, ~np.eye(N,dtype=bool)]<0).mean():.2f}')
    return video.astype(np.float32)

vol_video = build_corr_video(vol_sig, 'VOLUME')
rng_video = build_corr_video(rng_sig, 'VOLATILITY')

# ── Save ──────────────────────────────────────────────────────────────────────
p_vol = os.path.join(DATA, 'volume_corr_video.npy')
p_rng = os.path.join(DATA, 'volat_corr_video.npy')
np.save(p_vol, vol_video)
np.save(p_rng, rng_video)
print(f'\nSaved {p_vol}  ({os.path.getsize(p_vol)/1e6:.0f} MB)')
print(f'Saved {p_rng}  ({os.path.getsize(p_rng)/1e6:.0f} MB)')

# ── Cross-check independence vs return correlation (corr_video) ───────────────
cv = np.load(os.path.join(DATA, 'corr_video.npy'))
off = ~np.eye(N, dtype=bool)
def flat(v): return v[:, off].reshape(-1)
r_vol = np.corrcoef(flat(cv), flat(vol_video))[0, 1]
r_rng = np.corrcoef(flat(cv), flat(rng_video))[0, 1]
print(f'\nIndependence check (corr of off-diagonal entries vs return-corr video):')
print(f'  return-corr  vs  volume-corr     : {r_vol:+.3f}')
print(f'  return-corr  vs  volatility-corr : {r_rng:+.3f}')
print('  (low magnitude = genuinely new information, as the sanity check predicted)')
