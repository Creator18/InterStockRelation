"""Build systemic_anomaly.ipynb - does structural fragility precede market stress?

Uses the Absorption Ratio (Kritzman, Li, Page & Rigobon 2010): the fraction of
total variance captured by the top eigenvectors of the correlation matrix. A
high / rising AR means the market is tightly coupled and fragile; the paper shows
AR spikes precede drawdowns. This is a DIFFERENT target from our failed tests:
predict market-wide STRESS (drawdown / volatility), not correlation direction.

Tests:
  * AR and standardized dAR vs future H-day market return, drawdown, volatility
  * quantile analysis: future outcomes after high-dAR vs low-dAR days
  * incremental over VIX (does structure add beyond implied vol?)
No training.
"""
import json
def code(cid,src): return {"cell_type":"code","id":cid,"metadata":{},"execution_count":None,"outputs":[],"source":src.splitlines(keepends=True)}
def md(cid,src): return {"cell_type":"markdown","id":cid,"metadata":{},"source":src.splitlines(keepends=True)}

cells=[]
cells.append(md("md-01","""# Systemic Structural Anomaly: Does Fragility Precede Stress?

The **Absorption Ratio** (Kritzman et al., 2010) = fraction of total variance in
the correlation matrix captured by its top eigenvectors. High/rising AR = a
tightly-coupled, fragile market. Their finding: AR spikes *precede* market
drawdowns. We test this on our data --- a market-level early-warning, distinct
from the per-group onset that just failed.

Caveat up front: our sample is 2021--2026 and may not contain a major crash
(the COVID crash predates it), so the drawdown-prediction power of AR --- which
is strongest around crises --- may be muted or inconclusive here."""))

cells.append(code("cell-01","""# Cell 1 - Absorption Ratio series from corr_video + market returns from Nifty 50
import os, pickle, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from numpy.linalg import lstsq, eigvalsh

_cwd=os.getcwd()
ROOT=os.path.abspath(os.path.join(_cwd,'..')) if os.path.basename(_cwd)=='notebooks' else _cwd
DATA=os.path.join(ROOT,'data','processed'); RAW=os.path.join(ROOT,'data','raw'); PLOTS=os.path.join(ROOT,'outputs','plots')
corr=np.load(os.path.join(DATA,'corr_video.npy'))
pm=pickle.load(open(os.path.join(DATA,'pair_meta.pkl'),'rb'))
vm=pickle.load(open(os.path.join(DATA,'video_meta.pkl'),'rb'))
ts=pm['timestamps']; snap_ts=np.asarray(vm['snap_timesteps'])
T,N,_=corr.shape
H=5; W=60; KEIG=max(1,N//5)     # top ~20% eigenvectors (Kritzman convention)

# Absorption ratio per snapshot (trace of corr = N since diag=1)
AR=np.array([eigvalsh(corr[t])[::-1][:KEIG].sum()/N for t in range(T)])
lam1=np.array([eigvalsh(corr[t])[-1]/N for t in range(T)])   # market-mode fraction
# standardized change in AR (Kritzman's delta-AR signal)
armean=pd.Series(AR).rolling(W).mean().shift(1).to_numpy()
arstd =pd.Series(AR).rolling(W).std().shift(1).to_numpy()
dAR=(AR-armean)/np.where(arstd>1e-9,arstd,np.nan)

# market return from Nifty 50 index, aligned to snapshots
nifty=pd.read_csv(os.path.join(RAW,'NIFTY 50_5minute.csv'),parse_dates=['date']).set_index('date')['close']
nclose=pd.Series(nifty.reindex(ts).to_numpy()[snap_ts]).ffill().bfill().to_numpy()
rmkt=np.r_[np.nan, np.log(nclose[1:]/nclose[:-1])]
# VIX
vix=pd.read_csv(os.path.join(RAW,'INDIA VIX_5minute.csv'),parse_dates=['date']).set_index('date')['close']
VIX=pd.Series(vix.reindex(ts).to_numpy()[snap_ts]).ffill().bfill().to_numpy()
print(f'AR: mean {np.nanmean(AR):.3f} range [{np.nanmin(AR):.3f},{np.nanmax(AR):.3f}] (top {KEIG} of {N} eigvecs)')
print(f'market-mode fraction (lambda1/N): mean {np.nanmean(lam1):.3f}')
"""))

cells.append(code("cell-02","""# Cell 2 - Future market stress targets over [t+1, t+H]
fut_ret=np.full(T,np.nan); fut_dd=np.full(T,np.nan); fut_vol=np.full(T,np.nan)
for t in range(T-H):
    r=rmkt[t+1:t+1+H]
    if np.isnan(r).any(): continue
    fut_ret[t]=r.sum()                       # H-day forward return
    cum=np.cumsum(r); fut_dd[t]=cum.min()    # worst drawdown along the path
    fut_vol[t]=r.std()                        # realized volatility
print('future H-day market outcomes:')
print(f'  return : mean {np.nanmean(fut_ret):+.4f}  std {np.nanstd(fut_ret):.4f}')
print(f'  maxDD  : mean {np.nanmean(fut_dd):+.4f}  min {np.nanmin(fut_dd):+.4f}')
print(f'  vol    : mean {np.nanmean(fut_vol):.4f}')
"""))

cells.append(code("cell-03","""# Cell 3 - Does AR / dAR predict future stress?
idx=np.arange(W+1, T-H)
def corr_at(x,y):
    m=~(np.isnan(x)|np.isnan(y)); return pearsonr(x[m],y[m])[0] if m.sum()>20 else np.nan
print('Correlation of structural fragility with FUTURE market stress:')
print(f'{\"signal\":<10}{\"fut return\":>12}{\"fut drawdown\":>14}{\"fut vol\":>10}')
print('-'*46)
for nm,sig in [('AR',AR),('dAR',dAR),('lambda1',lam1),('VIX',VIX)]:
    print(f'{nm:<10}{corr_at(sig[idx],fut_ret[idx]):>12.3f}{corr_at(sig[idx],fut_dd[idx]):>14.3f}{corr_at(sig[idx],fut_vol[idx]):>10.3f}')
print('\\n(expect: high fragility -> negative future return & drawdown, positive future vol)')

# Quantile: outcomes after high vs low dAR
d=dAR[idx]; fr=fut_ret[idx]; fd=fut_dd[idx]
hi=d>np.nanpercentile(d,80); lo=d<np.nanpercentile(d,20)
print(f'\\nAfter HIGH dAR (top 20%): mean fut return {np.nanmean(fr[hi]):+.4f}, mean drawdown {np.nanmean(fd[hi]):+.4f}')
print(f'After LOW  dAR (bot 20%): mean fut return {np.nanmean(fr[lo]):+.4f}, mean drawdown {np.nanmean(fd[lo]):+.4f}')

# Incremental over VIX for predicting future volatility
def r2(cols,y):
    A=np.column_stack(cols+[np.ones(len(y))]); m=~np.isnan(A).any(1)&~np.isnan(y)
    if m.sum()<30: return np.nan
    c,_,_,_=lstsq(A[m],y[m],rcond=None); p=A[m]@c
    return 1-np.sum((y[m]-p)**2)/np.sum((y[m]-y[m].mean())**2)
print(f'\\nFuture-vol R^2:  VIX only={r2([VIX[idx]],fut_vol[idx]):.3f} | VIX+AR={r2([VIX[idx],AR[idx]],fut_vol[idx]):.3f}')
print(f'Future-return R^2: VIX only={r2([VIX[idx]],fut_ret[idx]):.3f} | VIX+dAR={r2([VIX[idx],dAR[idx]],fut_ret[idx]):.3f}')
"""))

cells.append(code("cell-04","""# Cell 4 - Plot + verdict
snap_dates=pd.to_datetime(pd.Series(vm['snap_dates']))
fig,ax=plt.subplots(2,1,figsize=(14,7),sharex=True)
fig.suptitle('Absorption Ratio (structural fragility) vs market',fontweight='bold')
ax[0].plot(snap_dates,AR,lw=0.8,color='purple'); ax[0].set_ylabel('Absorption Ratio'); ax[0].grid(alpha=0.3)
ax[1].plot(snap_dates,np.nancumsum(np.nan_to_num(rmkt)),lw=0.8,color='steelblue'); ax[1].set_ylabel('Nifty cum. log-return'); ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(PLOTS,'systemic_anomaly.png'),dpi=150,bbox_inches='tight'); plt.show()

c_dd=corr_at(dAR[idx],fut_dd[idx]); c_vol=corr_at(AR[idx],fut_vol[idx])
addvol=r2([VIX[idx],AR[idx]],fut_vol[idx])-r2([VIX[idx]],fut_vol[idx])
print('VERDICT'); print('='*60)
print(f'dAR -> future drawdown corr: {c_dd:+.3f} (want negative)')
print(f'AR  -> future vol corr     : {c_vol:+.3f} (want positive)')
print(f'AR incremental over VIX for future vol (R^2): {addvol:+.3f}')
print()
if (c_dd<-0.12) or (addvol>0.02 and c_vol>0.12):
    print('-> Structural fragility carries EARLY-WARNING signal for market stress.')
    print('   Worth a systemic-risk gauge in the app (distinct from the correlation-')
    print('   direction ceiling).')
else:
    print('-> No usable systemic early-warning in this sample (likely because it')
    print('   contains no major crash; AR shines around crises). Fragility is worth')
    print('   showing as a DESCRIPTIVE gauge, not a predictor.')
"""))

nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.11.9"}},"nbformat":4,"nbformat_minor":5}
open("notebooks/systemic_anomaly.ipynb","w").write(json.dumps(nb,indent=1))
print("Wrote notebooks/systemic_anomaly.ipynb with",len(cells),"cells")
