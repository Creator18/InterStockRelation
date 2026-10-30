"""Build a compact intraday feature file for Trial 10 (looped transformer).

Output: data/processed/intraday_features.npz  (~100 MB, upload this to the cluster)
  X        (T, N, 3) float32  standardized [return, dlog volume, log(high/low)]
  ret      (T, N)    float32  raw within-day 5-min log returns (0 at day open)
  bar      (T,)      int16    bar-of-day index (0..74)
  day      (T,)      int32    day id
  tickers  (N,)
Standardization stats use the first 70% of bars only (no look-ahead).
"""
import os, pickle
import numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA = os.path.join(ROOT, 'data', 'processed'); RAW = os.path.join(ROOT, 'data', 'raw')
pm = pickle.load(open(os.path.join(DATA, 'pair_meta.pkl'), 'rb'))
vm = pickle.load(open(os.path.join(DATA, 'video_meta.pkl'), 'rb'))
ts = pd.DatetimeIndex(pm['timestamps']); tickers = list(vm['ordered_tickers']); N = len(tickers); T = len(ts)

C = np.empty((T, N)); V = np.empty((T, N)); HI = np.empty((T, N)); LO = np.empty((T, N))
for k, tk in enumerate(tickers):
    d = pd.read_csv(os.path.join(RAW, f'{tk}_5minute.csv'), parse_dates=['date']).set_index('date')
    d = d[~d.index.duplicated()].reindex(ts).ffill()
    C[:, k] = d['close']; V[:, k] = d['volume']; HI[:, k] = d['high']; LO[:, k] = d['low']

day_key = ts.normalize()
day = pd.factorize(day_key)[0].astype(np.int32)
first = np.r_[True, day[1:] != day[:-1]]
bar = (pd.Series(np.arange(T)) - pd.Series(np.arange(T)).where(first).ffill()).to_numpy().astype(np.int16)

ret = np.zeros((T, N)); ret[1:] = np.diff(np.log(C), axis=0); ret[first] = 0.0
dv = np.zeros((T, N)); dv[1:] = np.diff(np.log1p(V), axis=0); dv[first] = 0.0
rng = np.log(np.maximum(HI, 1e-9) / np.maximum(LO, 1e-9))
feats = np.stack([ret, dv, rng], -1)
feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

cut = int(0.7 * T)
mu = feats[:cut].mean(0, keepdims=True); sd = feats[:cut].std(0, keepdims=True) + 1e-8
X = np.clip((feats - mu) / sd, -8, 8).astype(np.float32)

out = os.path.join(DATA, 'intraday_features.npz')
np.savez_compressed(out, X=X, ret=np.nan_to_num(ret).astype(np.float32), bar=bar, day=day, tickers=np.array(tickers))
print(f'wrote {out}: T={T} N={N} days={day.max()+1} max bar={bar.max()}')
