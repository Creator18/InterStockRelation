"""Build volume_signal_check.ipynb - Option 4 sanity check.

Quick check: does volume-correlation carry information different from
return-correlation? If yes, building a multi-stream Trial 7 is worth it.
If they're nearly identical, volume adds nothing and we save the effort.
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

cells.append(md("md-01", """# Volume Signal Sanity Check

**Question:** Does volume-correlation between two stocks carry information that is *different* from their return-correlation? If yes, we should build Trial 7 with multiple correlation streams (return + volume + volatility). If no, volume adds nothing useful and we skip ahead.

**Method:**
1. Pick 6 representative pairs (same sector, cross-sector, sector-specific).
2. For each pair, compute rolling correlations of three signals: returns, log-volume, and intraday volatility (high-low range).
3. Plot the three "correlation time series" side by side.
4. Quantify: cross-correlation between the three signals per pair. If near 1.0, they're redundant. If 0.4-0.8, they carry independent information.

Runs in under a minute on CPU."""))

cells.append(code("cell-01", """# Cell 1 - Imports and Configuration
import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

_cwd        = os.getcwd()
PROJECT_DIR = os.path.abspath(os.path.join(_cwd, '..')) \\
              if os.path.basename(_cwd) == 'notebooks' else _cwd
RAW_DIR     = os.path.join(PROJECT_DIR, 'data', 'raw')
PLOTS_DIR   = os.path.join(PROJECT_DIR, 'outputs', 'plots')

ROLL_WINDOW = 60     # number of 5-min bars in the rolling window
                     # 60 * 5 = 300 min = full trading day at NSE (~6.25 hrs)

# Representative pairs - varied sector/cross-sector
PAIRS = [
    ('HDFCBANK',  'ICICIBANK'),    # same-sector: large private banks
    ('TCS',       'INFY'),         # same-sector: IT
    ('RELIANCE',  'ONGC'),         # same-sector: oil & gas
    ('HDFCBANK',  'TCS'),          # cross-sector: bank vs IT
    ('RELIANCE',  'TCS'),          # cross-sector: energy vs IT
    ('SUNPHARMA', 'CIPLA'),        # same-sector: pharma
]

print(f'Pairs to analyze: {len(PAIRS)}')
for a, b in PAIRS:
    print(f'  {a} <-> {b}')
print(f'Rolling window: {ROLL_WINDOW} bars (~{ROLL_WINDOW*5} min)')
"""))

cells.append(code("cell-02", """# Cell 2 - Load Raw OHLCV for Required Stocks

needed_tickers = sorted({t for pair in PAIRS for t in pair})
print(f'Loading {len(needed_tickers)} stocks from {RAW_DIR} ...')

stock_data = {}
for tk in needed_tickers:
    fn = os.path.join(RAW_DIR, f'{tk}_5minute.csv')
    if not os.path.exists(fn):
        print(f'  MISSING: {fn} - skipping')
        continue
    df = pd.read_csv(fn, parse_dates=['date'])
    df = df.set_index('date').sort_index()
    # Derived quantities
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['log_volume'] = np.log(df['volume'].clip(lower=1))
    df['hl_range']   = (df['high'] - df['low']) / df['close']   # intraday volatility proxy
    stock_data[tk] = df
    print(f'  {tk:>12}: {len(df):>7,} bars  ({df.index[0].date()} -> {df.index[-1].date()})')

# Keep only pairs where both stocks loaded
PAIRS = [(a, b) for a, b in PAIRS if a in stock_data and b in stock_data]
print(f'\\nUsable pairs: {len(PAIRS)}')
"""))

cells.append(code("cell-03", """# Cell 3 - Compute Three Rolling Correlation Streams Per Pair

def rolling_corr(s1, s2, win):
    \"\"\"Rolling Pearson correlation between two aligned series.\"\"\"
    return s1.rolling(win).corr(s2)

pair_streams = {}
for a, b in PAIRS:
    da, db = stock_data[a], stock_data[b]
    # Align on overlapping timestamps
    idx = da.index.intersection(db.index)
    if len(idx) < ROLL_WINDOW * 2:
        print(f'  {a}<->{b}: only {len(idx)} overlapping bars - skipping')
        continue
    da_a = da.loc[idx]
    db_a = db.loc[idx]

    ret_corr = rolling_corr(da_a['log_return'], db_a['log_return'], ROLL_WINDOW)
    vol_corr = rolling_corr(da_a['log_volume'], db_a['log_volume'], ROLL_WINDOW)
    rng_corr = rolling_corr(da_a['hl_range'],   db_a['hl_range'],   ROLL_WINDOW)

    out = pd.DataFrame({
        'return_corr':     ret_corr,
        'volume_corr':     vol_corr,
        'volatility_corr': rng_corr,
    }).dropna()
    pair_streams[(a, b)] = out
    print(f'  {a:>11} <-> {b:<11}: {len(out):>7,} valid windows')
"""))

cells.append(code("cell-04", """# Cell 4 - Quantify Redundancy
# For each pair, compute the cross-correlation between the three streams.
# Near 1.0 means redundant. Lower means each stream carries independent info.

print(f'{\"Pair\":<28} {\"ret-vol\":>10} {\"ret-rng\":>10} {\"vol-rng\":>10}   verdict')
print('-' * 78)

redund_rows = []
for (a, b), df in pair_streams.items():
    r_rv, _ = pearsonr(df['return_corr'],     df['volume_corr'])
    r_rR, _ = pearsonr(df['return_corr'],     df['volatility_corr'])
    r_vR, _ = pearsonr(df['volume_corr'],     df['volatility_corr'])

    # Verdict: max of inter-stream correlation
    mx = max(abs(r_rv), abs(r_rR), abs(r_vR))
    if mx > 0.85:
        v = 'REDUNDANT'
    elif mx > 0.6:
        v = 'partial overlap'
    else:
        v = 'INDEPENDENT'

    name = f'{a} <-> {b}'
    print(f'{name:<28} {r_rv:>+10.4f} {r_rR:>+10.4f} {r_vR:>+10.4f}   {v}')
    redund_rows.append(dict(pair=name, ret_vol=r_rv, ret_rng=r_rR, vol_rng=r_vR, verdict=v))

redund_df = pd.DataFrame(redund_rows)
"""))

cells.append(code("cell-05", """# Cell 5 - Visualize: 6 pairs x 1 plot each, three streams overlaid

n_pairs = len(pair_streams)
fig, axes = plt.subplots(n_pairs, 1, figsize=(13, 2.5 * n_pairs),
                         sharex=True)
if n_pairs == 1:
    axes = [axes]
fig.suptitle(f'Rolling Correlations (window = {ROLL_WINDOW} bars / ~{ROLL_WINDOW*5} min)\\n'
             f'Are volume and volatility correlations the same as return correlation?',
             fontsize=13, fontweight='bold')

for ax, ((a, b), df) in zip(axes, pair_streams.items()):
    # Downsample for plot speed if needed
    plot_df = df.iloc[::10] if len(df) > 5000 else df
    ax.plot(plot_df.index, plot_df['return_corr'],     label='return corr',     color='steelblue',  alpha=0.85, lw=1)
    ax.plot(plot_df.index, plot_df['volume_corr'],     label='volume corr',     color='darkorange', alpha=0.85, lw=1)
    ax.plot(plot_df.index, plot_df['volatility_corr'], label='volatility corr', color='seagreen',   alpha=0.85, lw=1)
    ax.axhline(0, color='grey', linestyle=':', alpha=0.5)
    ax.set_title(f'{a} <-> {b}', fontsize=11)
    ax.set_ylabel('Rolling corr')
    ax.set_ylim(-1.1, 1.1)
    ax.legend(fontsize=8, loc='lower left')
    ax.grid(alpha=0.25)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'volume_signal_check.png'),
            dpi=120, bbox_inches='tight')
plt.show()
print(f'Saved: {os.path.join(PLOTS_DIR, \"volume_signal_check.png\")}')
"""))

cells.append(code("cell-06", """# Cell 6 - Verdict & Recommendation

n_independent  = (redund_df['verdict'] == 'INDEPENDENT').sum()
n_partial      = (redund_df['verdict'] == 'partial overlap').sum()
n_redundant    = (redund_df['verdict'] == 'REDUNDANT').sum()

print('VERDICT')
print('='*60)
print(f'  Independent streams : {n_independent}/{len(redund_df)} pairs')
print(f'  Partial overlap     : {n_partial}/{len(redund_df)} pairs')
print(f'  Redundant           : {n_redundant}/{len(redund_df)} pairs')
print()

avg_ret_vol = redund_df['ret_vol'].mean()
avg_ret_rng = redund_df['ret_rng'].mean()
avg_vol_rng = redund_df['vol_rng'].mean()
print(f'Average cross-stream correlations:')
print(f'  return  vs volume     : {avg_ret_vol:+.4f}')
print(f'  return  vs volatility : {avg_ret_rng:+.4f}')
print(f'  volume  vs volatility : {avg_vol_rng:+.4f}')
print()

if n_independent + n_partial >= len(redund_df) / 2:
    print('RECOMMENDATION: BUILD TRIAL 7.')
    print('  Volume and/or volatility carry information distinct from return correlation.')
    print('  A multi-channel pair-pixel architecture should be able to exploit this.')
elif n_redundant >= len(redund_df) - 1:
    print('RECOMMENDATION: SKIP volume/volatility streams.')
    print('  All three correlation types move together. Adding them as channels would')
    print('  just give the model duplicated information. Look elsewhere (Tier 2 macro).')
else:
    print('RECOMMENDATION: BUILD A TARGETED VERSION OF TRIAL 7.')
    print('  Mixed signal across pairs. Include volume/volatility streams but use')
    print('  attention or gating to let the model decide when to rely on each.')
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

with open("notebooks/volume_signal_check.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print(f'Wrote notebooks/volume_signal_check.ipynb with {len(cells)} cells')
