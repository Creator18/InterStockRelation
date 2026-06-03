"""Build ablation.ipynb — horizon ablation analysis WITHOUT training.

Maps the predictability of correlation videos as a function of horizon H
using baselines and information-theoretic measures. Tells us how much signal
exists at each horizon and where it decays — without any GPU/training cost.
"""
import json

def code(cid, src):
    return {"cell_type":"code","id":cid,"metadata":{},
            "execution_count":None,"outputs":[],
            "source":src.splitlines(keepends=True)}

def md(cid, src):
    return {"cell_type":"markdown","id":cid,"metadata":{},
            "source":src.splitlines(keepends=True)}

cells = []

cells.append(md("md-01", """# Horizon Ablation — No Training

**Question:** how much predictable signal lives in correlation videos at each forecast horizon?

**Approach:** characterise the predictability of `corr_video` using baselines and statistics, varied across horizon H ∈ {1, 2, 3, 5, 10, 20, 50}. No model training — every quantity here is either a closed-form statistic or a non-learning baseline.

**What we'll learn:**
1. **Persistence MSE curve** — how fast do correlations drift over H steps?
2. **Mean-frame MSE curve** — how does a naive "just predict the average" baseline scale?
3. **Trend-extrapolation baseline** — what does simple momentum give us?
4. **Directional accuracy ceiling from trend-following** — upper bound on dir_acc without learning
5. **Autocorrelation of frame deltas** — information half-life of the change signal
6. **Per-pair predictability distribution** — are a few pairs doing all the work?

**Why this matters:** our learned models plateau at val_mse ≈ 0.069, dir_acc ≈ 73% across many architectures. We need to know whether that's the *model's* fault or the *data's* fault. This ablation answers that."""))

cells.append(code("cell-01", """# Cell 1 – Imports & Load Data
import os, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

_cwd        = os.getcwd()
PROJECT_DIR = os.path.abspath(os.path.join(_cwd, '..')) \\
              if os.path.basename(_cwd) == 'notebooks' else _cwd
DATA_DIR    = os.path.join(PROJECT_DIR, 'data', 'processed')
PLOTS_DIR   = os.path.join(PROJECT_DIR, 'outputs', 'plots')

corr_video = np.load(os.path.join(DATA_DIR, 'corr_video.npy'))
with open(os.path.join(DATA_DIR, 'pair_meta.pkl'), 'rb') as f:
    pair_meta = pickle.load(f)
tickers = pair_meta['tickers']

T, N, _ = corr_video.shape
print(f'corr_video: {corr_video.shape}  (T={T} snapshots, N={N} stocks)')

# Same train/val split as the trials so results are comparable
LOOKBACK    = 20
TRAIN_RATIO = 0.70

HORIZONS = [1, 2, 3, 5, 10, 20, 50]
print(f'Horizons to test: {HORIZONS}')
"""))

cells.append(code("cell-02", """# Cell 2 – Reproduce the Same Train/Val Split Used in Trials 2-6
# So our baselines are evaluated on the SAME val snapshots the models saw.

# Use H=5 (the trial horizon) for the split construction, then re-evaluate on all H.
H_SPLIT = 5
n_valid = T - max(HORIZONS)        # ensure room for the longest horizon
frame_mse = np.array([
    float(np.mean((corr_video[t] - corr_video[t + H_SPLIT]) ** 2))
    for t in range(n_valid)
], dtype=np.float32)

volatility_threshold = float(np.median(frame_mse))
valid_snaps  = np.array([s for s in range(n_valid) if s >= LOOKBACK])
snap_v       = frame_mse[valid_snaps]
volatile_snaps = valid_snaps[snap_v >= volatility_threshold]
stable_snaps   = valid_snaps[snap_v <  volatility_threshold]

np.random.seed(42)
vol_train = np.random.choice(volatile_snaps, int(len(volatile_snaps) * TRAIN_RATIO), replace=False)
vol_val   = np.setdiff1d(volatile_snaps, vol_train)
sta_train = np.random.choice(stable_snaps,  int(len(stable_snaps)  * TRAIN_RATIO), replace=False)
sta_val   = np.setdiff1d(stable_snaps, sta_train)

train_snaps = np.sort(np.concatenate([vol_train, sta_train]))
val_snaps   = np.sort(np.concatenate([vol_val,   sta_val]))

mean_train_frame = corr_video[train_snaps].mean(axis=0)
print(f'Train: {len(train_snaps)}  Val: {len(val_snaps)}')
print(f'(Same stratified split used in Trials 2-6, so baselines are directly comparable.)')
"""))

cells.append(code("cell-03", """# Cell 3 – Baseline MSE Curves Across Horizons
# Three baselines per horizon:
#   1. Persistence:   predict corr[s+H] = corr[s]   (no change assumed)
#   2. Mean-frame:    predict corr[s+H] = mean of training frames
#   3. Trend extrap:  predict corr[s+H] = corr[s] + H * (corr[s] - corr[s-K]) / K
#                     using a short window K (captures local momentum)

K_TREND = 3   # use 3-step trend

def compute_baselines(H, K=K_TREND):
    # Only use val snapshots where we have both the lookback for trend AND the future for H
    sn = np.array([s for s in val_snaps if s + H < T and s - K >= 0])

    # 1. Persistence
    persist_mse = np.mean([
        np.mean((corr_video[s] - corr_video[s + H]) ** 2) for s in sn
    ])

    # 2. Mean-frame
    mean_mse = np.mean([
        np.mean((mean_train_frame - corr_video[s + H]) ** 2) for s in sn
    ])

    # 3. Trend extrapolation
    trend_mse = np.mean([
        np.mean((corr_video[s] + H * (corr_video[s] - corr_video[s - K]) / K
                 - corr_video[s + H]) ** 2)
        for s in sn
    ])

    return dict(n=len(sn), persistence=persist_mse, mean_frame=mean_mse,
                trend=trend_mse)

rows = []
for H in HORIZONS:
    b = compute_baselines(H)
    rows.append(dict(H=H, n=b['n'],
                     persistence=b['persistence'],
                     mean_frame=b['mean_frame'],
                     trend=b['trend'],
                     best_of_three=min(b['persistence'], b['mean_frame'], b['trend'])))

baseline_df = pd.DataFrame(rows)
print(baseline_df.to_string(index=False, float_format='%.6f'))
"""))

cells.append(code("cell-04", """# Cell 4 – Directional Accuracy Baselines
# Persistence-based dir_acc is meaningless (no predicted change).
# Two informative baselines:
#   A. Last-delta: predicted direction = sign of (corr[s] - corr[s-1])
#   B. Trend (K-step): predicted direction = sign of (corr[s] - corr[s-K])
# Compare both vs the true direction at horizon H: sign(corr[s+H] - corr[s])

def dir_acc_baselines(H, K=K_TREND, mag_thresh=0.005):
    sn = np.array([s for s in val_snaps if s + H < T and s - K >= 0])

    last_correct, trend_correct, masked_n = 0, 0, 0
    last_correct_masked, trend_correct_masked = 0, 0
    total_n = 0
    for s in sn:
        true_delta = corr_video[s + H] - corr_video[s]
        last_delta = corr_video[s]     - corr_video[s - 1]
        trend_delt = corr_video[s]     - corr_video[s - K]

        # All-cell directional accuracy
        last_correct  += np.sum(np.sign(last_delta) == np.sign(true_delta))
        trend_correct += np.sum(np.sign(trend_delt) == np.sign(true_delta))
        total_n       += true_delta.size

        # Magnitude-masked (only count cells where true change is non-trivial)
        mask = (np.abs(true_delta) > mag_thresh)
        masked_n += mask.sum()
        last_correct_masked  += np.sum((np.sign(last_delta) == np.sign(true_delta)) & mask)
        trend_correct_masked += np.sum((np.sign(trend_delt) == np.sign(true_delta)) & mask)

    return dict(
        n_samples       = len(sn),
        last_dir        = last_correct  / max(1, total_n),
        trend_dir       = trend_correct / max(1, total_n),
        last_dir_masked = last_correct_masked  / max(1, masked_n),
        trend_dir_masked= trend_correct_masked / max(1, masked_n),
    )

dir_rows = []
for H in HORIZONS:
    d = dir_acc_baselines(H)
    dir_rows.append(dict(H=H,
                         last_dir=d['last_dir'],
                         trend_dir=d['trend_dir'],
                         last_dir_masked=d['last_dir_masked'],
                         trend_dir_masked=d['trend_dir_masked']))

dir_df = pd.DataFrame(dir_rows)
print(dir_df.to_string(index=False, float_format='%.4f'))
print('\\n(last_dir = sign of 1-step delta as predictor;  trend_dir = sign of 3-step trend)')
print('(masked versions ignore cells where true |delta| <= 0.005)')
"""))

cells.append(code("cell-05", """# Cell 5 – Autocorrelation of Frame Deltas (Information Half-Life)
# For each pair (i,j), the time series of single-step deltas delta[t] = corr[t]-corr[t-1].
# autocorr(delta, H) tells us: do today's changes predict changes H steps from now?
#   > 0 : trend persists (momentum)
#   = 0 : random walk (no predictability beyond mean)
#   < 0 : mean reversion (overshoot)

delta_video = np.diff(corr_video, axis=0)       # (T-1, N, N)
mean_delta  = delta_video.mean(axis=0)
delta_demean = delta_video - mean_delta

# Compute per-pair autocorrelation at each lag, then average across pairs
# (averaging keeps signal even when individual pairs are noisy)
var_per_pair = (delta_demean ** 2).mean(axis=0) + 1e-10   # (N, N)

autocorr_results = []
for H in HORIZONS:
    cov = (delta_demean[:-H] * delta_demean[H:]).mean(axis=0)   # (N, N)
    autocorr = cov / var_per_pair
    autocorr_results.append(dict(
        H=H,
        mean_autocorr=float(autocorr.mean()),
        median_autocorr=float(np.median(autocorr)),
        frac_positive=float((autocorr > 0).mean()),
        frac_strong=float((autocorr.abs() > 0.05).mean()),
    ))
ac_df = pd.DataFrame(autocorr_results)
print(ac_df.to_string(index=False, float_format='%.4f'))
print('\\nReading guide:')
print('  mean_autocorr ≈ 0   → information is gone, no predictability from past changes')
print('  mean_autocorr > 0   → momentum present (trends persist)')
print('  mean_autocorr < 0   → mean reversion')
"""))

cells.append(code("cell-06", """# Cell 6 – Per-Pair Predictability Distribution
# Are a few pairs doing all the lifting? Histogram how predictable each pair is.

# Predictability proxy per pair: variance of its correlation series.
# A pair whose correlation barely moves is "easy" - persistence already nails it.
# A pair whose correlation swings wildly is "hard".

pair_var = corr_video.var(axis=0)                  # (N, N)
pair_var_flat = pair_var[np.triu_indices(N, k=1)]  # unique pairs only

# Persistence error per pair at H=5
H_REF = 5
persist_err = np.array([
    ((corr_video[s] - corr_video[s + H_REF]) ** 2) for s in val_snaps if s + H_REF < T
]).mean(axis=0)                                    # (N, N) MSE per pair
persist_err_flat = persist_err[np.triu_indices(N, k=1)]

# Mean-frame error per pair at H=5
mean_err = np.array([
    ((mean_train_frame - corr_video[s + H_REF]) ** 2) for s in val_snaps if s + H_REF < T
]).mean(axis=0)
mean_err_flat = mean_err[np.triu_indices(N, k=1)]

print(f'Per-pair MSE at H={H_REF}  (N_pairs={len(pair_var_flat)})')
print(f'  Persistence: median={np.median(persist_err_flat):.5f}  '
      f'95th pct={np.percentile(persist_err_flat, 95):.5f}')
print(f'  Mean-frame : median={np.median(mean_err_flat):.5f}    '
      f'95th pct={np.percentile(mean_err_flat, 95):.5f}')

# How concentrated is the error? Top-5% of pairs contribute what fraction of total MSE?
n_top = int(0.05 * len(persist_err_flat))
top5_persist = np.sort(persist_err_flat)[-n_top:].sum() / persist_err_flat.sum()
top5_mean    = np.sort(mean_err_flat   )[-n_top:].sum() / mean_err_flat   .sum()
print(f'\\nConcentration: top 5% of pairs account for ...')
print(f'  {top5_persist*100:.1f}% of persistence MSE')
print(f'  {top5_mean*100:.1f}% of mean-frame MSE')
print('\\n(If a small fraction of pairs dominates the error, focused work on those pairs would help.)')
"""))

cells.append(code("cell-07", """# Cell 7 – Reference Trial 5 Performance Against Baselines

# Trial 5 ConvLSTM final numbers, evaluated at H=5
T5_MSE   = 0.068600
T5_DIR   = 0.730
T5_PEAR  = 0.3724

H = 5
b5  = compute_baselines(H)
d5  = dir_acc_baselines(H)

print(f'COMPARISON AT H={H} (where Trial 5 model was trained)')
print('='*72)
print(f'{"Method":<32} {"MSE":>10} {"Dir Acc":>10} {"vs Persist":>12}')
print('-'*72)
def show(name, mse, dir_acc, base=b5['persistence']):
    vs = (base - mse) / base * 100 if mse is not None else None
    mse_s = f'{mse:.6f}' if mse is not None else '   -    '
    dir_s = f'{dir_acc:.4f}' if dir_acc is not None else '  -   '
    vs_s  = f'{vs:+.1f}%' if vs is not None else '   -  '
    print(f'{name:<32} {mse_s:>10} {dir_s:>10} {vs_s:>12}')

show('Persistence',           b5['persistence'], 0.5)
show('Mean-frame',             b5['mean_frame'],  None)
show('Trend extrapolation',    b5['trend'],       d5['trend_dir'])
show('Last-delta dir baseline', None,             d5['last_dir'])
show('Last-delta dir (masked)', None,             d5['last_dir_masked'])
print('-'*72)
show('Trial 5 ConvLSTM (learned)', T5_MSE, T5_DIR)
print()
print(f'Headroom analysis:')
print(f'  Mean-frame to T5:   {b5["mean_frame"] - T5_MSE:+.6f}  ({(b5["mean_frame"] - T5_MSE)/b5["mean_frame"]*100:+.1f}% of mean-frame MSE)')
print(f'  T5 dir vs trend dir: {(T5_DIR - d5["trend_dir"])*100:+.1f}pp added by learning beyond simple trend-follow')
print(f'  T5 dir vs last-delta dir: {(T5_DIR - d5["last_dir"])*100:+.1f}pp added by learning beyond 1-step trend')
"""))

cells.append(code("cell-08", """# Cell 8 – Plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Horizon Ablation — Predictability of Correlation Videos',
             fontsize=14, fontweight='bold')

# (1) MSE baselines vs H
ax = axes[0, 0]
ax.plot(baseline_df['H'], baseline_df['persistence'], 'o-', label='Persistence (predict no change)',
        color='steelblue')
ax.plot(baseline_df['H'], baseline_df['mean_frame'], 's-', label='Mean-frame (predict avg)',
        color='darkorange')
ax.plot(baseline_df['H'], baseline_df['trend'],      '^-', label='Trend extrap (K=3)',
        color='seagreen')
ax.axhline(T5_MSE, color='crimson', linestyle='--',
           label=f'Trial 5 ConvLSTM (H=5)')
ax.set_xlabel('Horizon H'); ax.set_ylabel('Val MSE')
ax.set_title('MSE baselines vs horizon')
ax.set_xscale('log'); ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# (2) Directional baselines vs H
ax = axes[0, 1]
ax.plot(dir_df['H'], dir_df['last_dir'],         'o-', label='1-step trend (all cells)',     color='steelblue')
ax.plot(dir_df['H'], dir_df['trend_dir'],        's-', label='3-step trend (all cells)',     color='darkorange')
ax.plot(dir_df['H'], dir_df['last_dir_masked'],  'o--', label='1-step trend (|Δ|>0.005)',     color='steelblue', alpha=0.6)
ax.plot(dir_df['H'], dir_df['trend_dir_masked'], 's--', label='3-step trend (|Δ|>0.005)',     color='darkorange', alpha=0.6)
ax.axhline(T5_DIR, color='crimson', linestyle='--', label=f'Trial 5 dir_acc (H=5)')
ax.axhline(0.5,    color='grey',    linestyle=':',  label='Random (0.5)')
ax.set_xlabel('Horizon H'); ax.set_ylabel('Directional Accuracy')
ax.set_title('Directional-accuracy baselines vs horizon')
ax.set_xscale('log'); ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# (3) Autocorrelation decay
ax = axes[1, 0]
ax.plot(ac_df['H'], ac_df['mean_autocorr'],   'o-', label='Mean autocorr',   color='steelblue')
ax.plot(ac_df['H'], ac_df['median_autocorr'], 's-', label='Median autocorr', color='darkorange')
ax.axhline(0.0, color='grey', linestyle=':')
ax.set_xlabel('Horizon H'); ax.set_ylabel('Autocorrelation of frame-deltas')
ax.set_title('Information half-life of correlation changes')
ax.set_xscale('log'); ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# (4) Per-pair MSE distribution
ax = axes[1, 1]
ax.hist(persist_err_flat, bins=80, alpha=0.5, label='Persistence  per-pair MSE @ H=5', color='steelblue')
ax.hist(mean_err_flat,    bins=80, alpha=0.5, label='Mean-frame  per-pair MSE @ H=5', color='darkorange')
ax.set_yscale('log')
ax.set_xlabel('Per-pair MSE'); ax.set_ylabel('Number of pairs')
ax.set_title('Predictability is highly heterogeneous across pairs')
ax.legend(fontsize=9); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'horizon_ablation.png'), dpi=150, bbox_inches='tight')
plt.show()

print(f'Saved {os.path.join(PLOTS_DIR, "horizon_ablation.png")}')
"""))

cells.append(code("cell-09", """# Cell 9 – Summary & Interpretation

# Find the crossover where persistence stops being better than mean-frame.
# Beyond that horizon, we're predicting averages, not dynamics.
crossover = None
for i, row in baseline_df.iterrows():
    if row['persistence'] > row['mean_frame']:
        crossover = row['H']
        break

print('SUMMARY')
print('='*70)
print()
print(f'Persistence/Mean-frame crossover horizon: H ≈ {crossover}')
print(f'  → beyond H={crossover}, "predict no change" is worse than "predict average"')
print(f'  → useful prediction window is roughly H < {crossover}')
print()
print(f'Information half-life (mean autocorr drops below 0.05):')
ac_lt = ac_df[ac_df['mean_autocorr'].abs() < 0.05]
if len(ac_lt) > 0:
    print(f'  → at H ≈ {ac_lt.iloc[0]["H"]}  (autocorr = {ac_lt.iloc[0]["mean_autocorr"]:.4f})')
else:
    print(f'  → not reached within tested horizons; signal persists longer than H={HORIZONS[-1]}')
print()
print('At Trial 5 horizon (H=5):')
b5 = baseline_df[baseline_df['H'] == 5].iloc[0]
d5 = dir_df     [dir_df['H']      == 5].iloc[0]
gap_to_optimal = b5['mean_frame'] - T5_MSE
useful_room    = b5['mean_frame'] - b5['best_of_three']
print(f'  MSE gap mean_frame → trial5    : {gap_to_optimal:.6f}')
print(f'  MSE gap mean_frame → best base : {useful_room:.6f}  (without learning)')
print(f'  Headroom left for learning     : {useful_room - gap_to_optimal:.6f}  ({(useful_room - gap_to_optimal)/b5["mean_frame"]*100:+.2f}% of mean-frame MSE)')
print()
print(f'  Dir acc — random                : 0.500')
print(f'  Dir acc — 1-step trend (masked) : {d5["last_dir_masked"]:.4f}')
print(f'  Dir acc — 3-step trend (masked) : {d5["trend_dir_masked"]:.4f}')
print(f'  Dir acc — Trial 5 ConvLSTM      : {T5_DIR:.4f}')
print(f'  Gain from learning vs trend     : {(T5_DIR - d5["trend_dir_masked"])*100:+.1f}pp')
print()
print('INTERPRETATION:')
print('-'*70)
short_hzn_gap = baseline_df.iloc[0]['mean_frame'] - baseline_df.iloc[0]['best_of_three']
long_hzn_gap  = baseline_df.iloc[-1]['mean_frame'] - baseline_df.iloc[-1]['best_of_three']
print(f'  Predictability at H=1:  baseline can close {short_hzn_gap:.5f} of MSE gap')
print(f'  Predictability at H={HORIZONS[-1]}:  baseline can close {long_hzn_gap:.5f} of MSE gap')
"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.9"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("notebooks/ablation.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print(f'Wrote notebooks/ablation.ipynb with {len(cells)} cells')
