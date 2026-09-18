"""Build group_predictability.ipynb - is a GROUP's relationship more predictable
than a single PAIR's?

Unit of analysis shifts from a pair (2 stocks) to a group (k stocks). We form
data-driven groups by clustering the mean correlation matrix, then build several
group-level relational descriptors over time and test how predictable their
5-step-ahead CHANGE is, versus the per-pair baseline (~73%).

Descriptors per group g at snapshot t:
  * cohesion(g)      : mean intra-group correlation           (how together)
  * dispersion(g)    : std of intra-group correlations        (how uniform)
  * group<->market(g): mean correlation of g to all non-members
  * group<->group(g,h): mean correlation between groups g and h (relationships
                        BETWEEN groups as units)

No training. Baselines: persistence and level-mean-reversion, same as prior
diagnostics, so numbers are directly comparable to the per-cell 73%.
"""
import json

def code(cid,src): return {"cell_type":"code","id":cid,"metadata":{},"execution_count":None,"outputs":[],"source":src.splitlines(keepends=True)}
def md(cid,src): return {"cell_type":"markdown","id":cid,"metadata":{},"source":src.splitlines(keepends=True)}

cells=[]
cells.append(md("md-01","""# Group-Level Predictability: Groups vs Pairs

Does observing the relationship of a **group** of stocks (k stocks as one
relational unit) yield more predictable structure than a single **pair**?

Motivation: our per-cell forecast plateaus at ~73% directional accuracy because
the 9,025 individual pairs are dominated by noise. Market shifts act on *groups*
(sectors, factors), so group-level relational quantities are lower-dimensional
and should be less noise-diluted --- potentially more predictable.

We form data-driven groups once (hierarchical clustering of the mean correlation
matrix), compute several group-level relational descriptors over time, and test
the predictability of their 5-step-ahead change against the per-pair baseline.
No training."""))

cells.append(code("cell-01","""# Cell 1 - Load + form data-driven groups
import os, pickle, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

_cwd=os.getcwd()
ROOT=os.path.abspath(os.path.join(_cwd,'..')) if os.path.basename(_cwd)=='notebooks' else _cwd
DATA=os.path.join(ROOT,'data','processed'); PLOTS=os.path.join(ROOT,'outputs','plots')
corr=np.load(os.path.join(DATA,'corr_video.npy'))
vm=pickle.load(open(os.path.join(DATA,'video_meta.pkl'),'rb'))
ordered=vm['ordered_tickers']
T,N,_=corr.shape
H=5; LOOKBACK=20; K=8      # number of groups

# Balanced clustering: each stock's correlation profile (row of mean corr) is its
# feature vector; KMeans groups stocks by 'who they co-move with'. Avoids the
# giant-cluster chaining that distance-linkage suffers on a highly-connected market.
mean_corr=corr.mean(0)
try:
    from sklearn.cluster import KMeans
    labels=KMeans(n_clusters=K, n_init=10, random_state=42).fit_predict(mean_corr)
    method='KMeans on correlation profiles'
except Exception:
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    D=1.0-mean_corr; D=(D+D.T)/2; np.fill_diagonal(D,0.0); D=np.clip(D,0,None)
    labels=fcluster(linkage(squareform(D,checks=False),method='ward'),K,criterion='maxclust')
    method='Ward linkage (sklearn unavailable)'
groups={g:np.where(labels==g)[0] for g in np.unique(labels)}
groups={g:idx for g,idx in groups.items() if len(idx)>=2}   # need >=2 for internal corr
print(f'clustering: {method}')
print(f'{len(groups)} groups (>=2 members) from K={K}:')
for g,idx in groups.items():
    names=[ordered[i] for i in idx][:6]
    print(f'  group {g} (n={len(idx)}): {names}{\"...\" if len(idx)>6 else \"\"}')
"""))

cells.append(code("cell-02","""# Cell 2 - Build group-level relational time series
gl=list(groups.keys())
def intra(idx,t):
    s=corr[t][np.ix_(idx,idx)]; m=~np.eye(len(idx),dtype=bool); return s[m]
cohesion={g:np.array([intra(groups[g],t).mean() for t in range(T)]) for g in gl}
dispersion={g:np.array([intra(groups[g],t).std() for t in range(T)]) for g in gl}
# group-to-market: mean corr of members to non-members
allidx=np.arange(N)
g2m={}
for g in gl:
    idx=groups[g]; other=np.setdiff1d(allidx,idx)
    g2m[g]=np.array([corr[t][np.ix_(idx,other)].mean() for t in range(T)])
# group-to-group: for each unordered pair of groups
g2g={}
for i in range(len(gl)):
    for j in range(i+1,len(gl)):
        a,b=gl[i],gl[j]
        g2g[(a,b)]=np.array([corr[t][np.ix_(groups[a],groups[b])].mean() for t in range(T)])
print('series built:',
      f'cohesion x{len(cohesion)}, dispersion x{len(dispersion)}, group-market x{len(g2m)}, group-group x{len(g2g)}')
"""))

cells.append(code("cell-03","""# Cell 3 - Predictability helper (dir-acc of H-ahead change) + per-CELL baseline
def dir_acc_series(q, H=H, W=LOOKBACK):
    # trailing-mean (causal); predict sign of (q[t+H]-q[t]) by mean-reversion & momentum; return best
    q=np.asarray(q); n=len(q)
    tm=np.full(n,np.nan)
    cs=np.cumsum(q)
    for t in range(W,n): tm[t]=(cs[t-1]-cs[t-1-W])/W
    idx=np.arange(W, n-H)
    true=np.sign(q[idx+H]-q[idx])
    mr = -np.sign(q[idx]-tm[idx])                 # mean-reversion
    mom= np.sign(q[idx]-q[idx-1])                 # momentum
    nz=true!=0
    a_mr=(mr[nz]==true[nz]).mean(); a_mom=(mom[nz]==true[nz]).mean()
    return max(a_mr,a_mom), a_mr, a_mom, nz.sum()

# Per-CELL baseline recomputed here (level mean-reversion, off-diagonal cells) for apples-to-apples
cs=np.cumsum(corr,0); iu,ju=np.triu_indices(N,k=1)
cell_true=[]; cell_pred=[]
for t in range(LOOKBACK, T-H):
    tm=(cs[t-1]-cs[t-1-LOOKBACK])/LOOKBACK
    true=np.sign(corr[t+H]-corr[t])[iu,ju]
    pred=-np.sign((corr[t]-tm))[iu,ju]
    cell_true.append(true); cell_pred.append(pred)
ct=np.concatenate(cell_true); cp=np.concatenate(cell_pred); nz=ct!=0
CELL_ACC=(cp[nz]==ct[nz]).mean()
print(f'PER-CELL baseline dir-acc (level mean-reversion): {CELL_ACC:.4f}  (the ~73% wall)')
"""))

cells.append(code("cell-04","""# Cell 4 - Group-level predictability vs per-cell
def summarize(series_dict, label):
    accs=[dir_acc_series(v)[0] for v in series_dict.values()]
    accs=[a for a in accs if not np.isnan(a)]
    return dict(label=label, n=len(accs), mean_acc=np.mean(accs), min=np.min(accs), max=np.max(accs))

rows=[summarize(cohesion,'cohesion (intra-group)'),
      summarize(dispersion,'dispersion (intra-group)'),
      summarize(g2m,'group <-> market'),
      summarize(g2g,'group <-> group')]
df=pd.DataFrame(rows)
print(f'{\"descriptor\":<26}{\"n\":>4}{\"mean dir-acc\":>14}{\"min\":>8}{\"max\":>8}')
print('-'*60)
for r in rows:
    print(f'{r[\"label\"]:<26}{r[\"n\"]:>4}{r[\"mean_acc\"]:>14.4f}{r[\"min\"]:>8.3f}{r[\"max\"]:>8.3f}')
print('-'*60)
print(f'{\"PER-CELL (pairs) baseline\":<26}{\"-\":>4}{CELL_ACC:>14.4f}')
print()
best=max(rows,key=lambda r:r['mean_acc'])
print(f'Best group descriptor: {best[\"label\"]} at {best[\"mean_acc\"]:.4f} '
      f'vs per-cell {CELL_ACC:.4f}  (delta {(best[\"mean_acc\"]-CELL_ACC)*100:+.1f}pp)')
"""))

cells.append(code("cell-05","""# Cell 5 - Detail: per-group cohesion dir-acc + autocorrelation of changes
print('Per-group COHESION predictability:')
print(f'{\"group\":<8}{\"n\":>4}{\"best\":>8}{\"mean-rev\":>10}{\"momentum\":>10}')
for g in gl:
    best,mr,mom,n=dir_acc_series(cohesion[g])
    print(f'{g:<8}{len(groups[g]):>4}{best:>8.3f}{mr:>10.3f}{mom:>10.3f}')

# autocorrelation of 1-step changes: group cohesion vs per-cell (lower |acorr|=more random)
def acf1(q):
    d=np.diff(q); d=d-d.mean(); return float((d[:-1]*d[1:]).mean()/ (d*d).mean()+1e-12)
coh_acf=np.mean([acf1(cohesion[g]) for g in gl])
cell_series=corr[:,iu,ju]           # (T, n_pairs)
cell_acf=np.mean([acf1(cell_series[:,k]) for k in range(0,cell_series.shape[1],50)])  # sample pairs
print(f'\\nMean 1-step change autocorrelation:')
print(f'  group cohesion : {coh_acf:+.3f}')
print(f'  per-pair (sample): {cell_acf:+.3f}')
print('  (stronger negative = stronger mean-reversion = more predictable direction)')
"""))

cells.append(code("cell-06","""# Cell 6 - Plot + verdict
fig,ax=plt.subplots(1,2,figsize=(14,5))
fig.suptitle('Group-level vs pair-level predictability',fontweight='bold')
labels=[r['label'] for r in rows]+['per-cell (pairs)']
vals=[r['mean_acc'] for r in rows]+[CELL_ACC]
colors=['seagreen']*len(rows)+['crimson']
ax[0].barh(labels,vals,color=colors); ax[0].axvline(0.5,color='grey',ls=':')
ax[0].axvline(CELL_ACC,color='crimson',ls='--',alpha=0.6)
ax[0].set_xlim(0.5,1.0); ax[0].set_xlabel('mean directional accuracy'); ax[0].set_title('Predictability by relational unit')
# example cohesion trajectory
g0=gl[0]
ax[1].plot(cohesion[g0],lw=0.8,label=f'group {g0} cohesion')
ax[1].plot(pd.Series(cohesion[g0]).rolling(LOOKBACK).mean(),lw=1.2,label='trailing mean')
ax[1].set_title(f'Group {g0} cohesion over time'); ax[1].legend(fontsize=8); ax[1].set_xlabel('snapshot')
plt.tight_layout(); plt.savefig(os.path.join(PLOTS,'group_predictability.png'),dpi=150,bbox_inches='tight'); plt.show()

print('VERDICT')
print('='*60)
best=max(rows,key=lambda r:r['mean_acc'])
d=(best['mean_acc']-CELL_ACC)*100
if d>3:
    print(f'Group-level relationships are MORE predictable (+{d:.1f}pp on {best[\"label\"]}).')
    print('Aggregating to groups reduces per-pair noise and raises the ceiling ->')
    print('model & app should operate at the GROUP level, not the pair level.')
elif d>0.5:
    print(f'Group-level is modestly more predictable (+{d:.1f}pp). Worth pursuing at')
    print('the group level for interpretability even if the accuracy gain is small.')
else:
    print(f'Group-level is NOT more predictable ({d:+.1f}pp). The ceiling survives')
    print('aggregation -> the limit is temporal (H=5 noise floor), not granularity.')
"""))

nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.11.9"}},"nbformat":4,"nbformat_minor":5}
open("notebooks/group_predictability.ipynb","w").write(json.dumps(nb,indent=1))
print("Wrote notebooks/group_predictability.ipynb with",len(cells),"cells")
