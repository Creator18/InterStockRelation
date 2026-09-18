"""Build onset_detection.ipynb - does PARTIAL group activation precede FULL
group activation? (intra-group contagion onset / early-warning)

Hypothesis: a shock enters a market space through a few 'canary' stocks before
the whole group moves. If a few members activate today (but the group isn't yet
fully moving), is the rest of the group more likely to activate over the next H
days than when nobody moved?

Key control: we condition on the group NOT being fully active today, so we're
testing genuine ONSET from a quiet/partial state -- not 'active stays active'
(volatility clustering), which would be a trivial confound.

Outputs:
  * P(future full activation | k early movers today), bucketed, with lift vs base
  * continuous precursor: today's fraction-active -> future group intensity,
    partial-correlated controlling for today's intensity (removes vol clustering)
  * canary stocks: which members most often move BEFORE a group event
  * identification: each group labeled by its dominant market space
No training.
"""
import json

def code(cid,src): return {"cell_type":"code","id":cid,"metadata":{},"execution_count":None,"outputs":[],"source":src.splitlines(keepends=True)}
def md(cid,src): return {"cell_type":"markdown","id":cid,"metadata":{},"source":src.splitlines(keepends=True)}

cells=[]
cells.append(md("md-01","""# Onset Detection: Do Early Movers Precede a Group Event?

Tests the 'entry point' hypothesis: a market-space shock starts in a few member
stocks before the whole group is hit. We measure whether **partial activation**
today (a few members making outsized moves, while the group is *not* yet fully
active) predicts **full activation** of the group over the next H days --- above
the base rate and above what volatility-clustering alone would give.

If yes: an early-warning alert signal exists (the core of the app). If flat: in
this large-cap universe the onset is too fast to catch at daily resolution, and
we fall back to same-day detection + identification (which still works)."""))

cells.append(code("cell-01","""# Cell 1 - Load, form groups, returns, activation flags
import os, pickle, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

_cwd=os.getcwd()
ROOT=os.path.abspath(os.path.join(_cwd,'..')) if os.path.basename(_cwd)=='notebooks' else _cwd
DATA=os.path.join(ROOT,'data','processed'); RAW=os.path.join(ROOT,'data','raw'); PLOTS=os.path.join(ROOT,'outputs','plots')
corr=np.load(os.path.join(DATA,'corr_video.npy'))
pm=pickle.load(open(os.path.join(DATA,'pair_meta.pkl'),'rb'))
vm=pickle.load(open(os.path.join(DATA,'video_meta.pkl'),'rb'))
ts=pm['timestamps']; snap_ts=np.asarray(vm['snap_timesteps']); ordered=vm['ordered_tickers']
T,N,_=corr.shape

H=5; W=60; Z_THRESH=2.0; ACT_FRAC=0.40; K=8

# groups: KMeans on correlation profiles (balanced, sector-like)
from sklearn.cluster import KMeans
labels=KMeans(n_clusters=K,n_init=10,random_state=42).fit_predict(corr.mean(0))
groups={g:np.where(labels==g)[0] for g in np.unique(labels)}
groups={g:idx for g,idx in groups.items() if len(idx)>=3}

# per-stock snapshot returns (ordered/matrix positions) + causal z-scores
close=pd.DataFrame(index=ts)
for tk in ordered:
    d=pd.read_csv(os.path.join(RAW,f'{tk}_5minute.csv'),parse_dates=['date']).set_index('date')
    close[tk]=d['close'].reindex(ts)
sc=close.iloc[snap_ts].to_numpy()
ret=np.vstack([np.full((1,N),np.nan), np.log(sc[1:]/sc[:-1])])
tstd=pd.DataFrame(ret).rolling(W).std().shift(1).to_numpy()   # causal trailing std
z=ret/np.where(tstd>1e-9,tstd,np.nan)
activated=(np.abs(z)>Z_THRESH)                                 # (T,N) bool
print(f'{len(groups)} groups | H={H} W={W} |z|>{Z_THRESH} | full-activation = >={ACT_FRAC:.0%} of members')
for g,idx in groups.items():
    print(f'  group {g} (n={len(idx)}): {[ordered[i] for i in idx][:6]}...')
"""))

cells.append(code("cell-02","""# Cell 2 - ONSET TEST: P(future full activation | k early movers today, given not-full today)
frac_act={g:np.nanmean(activated[:,idx],axis=1) for g,idx in groups.items()}   # (T,) per group
kcount ={g:np.nansum(activated[:,idx],axis=1) for g,idx in groups.items()}
full   ={g:(frac_act[g]>=ACT_FRAC) for g in groups}

valid=np.arange(W+1, T-H)
buckets={'0':[], '1':[], '2':[], '3+':[]}
base_hits=0; base_n=0
for g in groups:
    fa=frac_act[g]; k=kcount[g]; fu=full[g]
    for t in valid:
        if fu[t]:                       # skip: group already fully active today
            continue
        future_full=bool(fu[t+1:t+1+H].any())
        base_hits+=future_full; base_n+=1
        kk=int(k[t])
        key='0' if kk==0 else '1' if kk==1 else '2' if kk==2 else '3+'
        buckets[key].append(future_full)
base=base_hits/base_n
print(f'Base rate  P(group fully activates within next {H} days | currently not full) = {base:.3f}')
print(f'(n={base_n} group-days)\\n')
print(f'{\"early movers today (k)\":<24}{\"n\":>8}{\"P(future full)\":>16}{\"lift\":>8}')
print('-'*58)
for key in ['0','1','2','3+']:
    b=buckets[key]
    if b: print(f'{key:<24}{len(b):>8}{np.mean(b):>16.3f}{np.mean(b)/base:>8.2f}x')
print('\\nONSET SIGNAL EXISTS if P(future full) RISES with k (esp. for k=1,2 while not-full today).')
"""))

cells.append(code("cell-03","""# Cell 3 - Continuous precursor (controls for volatility clustering) + canary stocks
from scipy.stats import pearsonr
from numpy.linalg import lstsq
# today's fraction-active -> future group intensity (mean |z| over next H), controlling for today's intensity
X=[]; Yf=[]; Yc=[]
for g in groups:
    idx=groups[g]; fa=frac_act[g]
    inten=np.nanmean(np.abs(z[:,idx]),axis=1)   # group activation intensity per day
    for t in np.arange(W+1,T-H):
        X.append(fa[t]); Yc.append(inten[t]); Yf.append(np.nanmean(inten[t+1:t+1+H]))
X=np.array(X); Yc=np.array(Yc); Yf=np.array(Yf)
m=~(np.isnan(X)|np.isnan(Yc)|np.isnan(Yf))
X,Yc,Yf=X[m],Yc[m],Yf[m]
raw=pearsonr(X,Yf)[0]
# partial: residualize future intensity on today's intensity, correlate with today's frac-active
b=np.polyfit(Yc,Yf,1); resid=Yf-np.polyval(b,Yc)
part=pearsonr(X,resid)[0]
print('Continuous precursor (today fraction-active -> future group intensity):')
print(f'  raw corr                    : {raw:+.3f}')
print(f'  partial (control today vol) : {part:+.3f}   <- the honest number')
print('  (raw is inflated by vol clustering; partial isolates genuine onset lead)\\n')

# Canary stocks: members most often activated the day BEFORE a group onset
print('Canary stocks (activated on the day a group onset begins), per group:')
for g in groups:
    idx=groups[g]; fu=full[g]
    onset=np.array([ (not fu[t]) and bool(fu[t+1:t+1+H].any()) for t in range(T)])  # not-full today, full soon
    cnt={}
    for t in np.where(onset)[0]:
        for i in idx:
            if activated[t,i]: cnt[i]=cnt.get(i,0)+1
    tot=onset.sum()
    top=sorted(cnt.items(),key=lambda x:-x[1])[:3]
    names=', '.join(f'{ordered[i]}({c/max(1,tot)*100:.0f}%)' for i,c in top)
    print(f'  group {g}: {names}')
"""))

cells.append(code("cell-04","""# Cell 4 - Identification: label each group's market space + plot onset curve
KNOWN={  # rough sector tags for labeling (identification half)
 'BANK':['HDFCBANK','ICICIBANK','SBIN','AXISBANK','KOTAKBANK','BANKBARODA','PNB','CANBK','UNIONBANK'],
 'FIN':['BAJFINANCE','BAJAJFINSV','CHOLAFIN','SHRIRAMFIN','MUTHOOTFIN','PFC','RECLTD','HDFCAMC','IRFC'],
 'IT':['TCS','INFY','WIPRO','HCLTECH','TECHM','LTM'],
 'PHARMA':['SUNPHARMA','CIPLA','DRREDDY','DIVISLAB','TORNTPHARM','ZYDUSLIFE','APOLLOHOSP','MAXHEALTH'],
 'ENERGY':['RELIANCE','ONGC','BPCL','IOC','GAIL','NTPC','POWERGRID','COALINDIA','TATAPOWER','ADANIPOWER','ADANIGREEN','ADANIENSOL'],
 'METAL':['TATASTEEL','JSWSTEEL','JINDALSTEL','HINDALCO','VEDL','HINDZINC'],
 'AUTO':['MARUTI','MM','TVSMOTOR','EICHERMOT','BAJAJ-AUTO','BOSCHLTD','MOTHERSON','TMPV'],
 'FMCG':['HINDUNILVR','ITC','NESTLEIND','BRITANNIA','GODREJCP','TATACONSUM','DMART','UNITDSPR','VBL'],
}
def label_group(idx):
    names=set(ordered[i] for i in idx); best,score=None,0
    for sec,mem in KNOWN.items():
        s=len(names & set(mem))
        if s>score: best,score=sec,s
    return f'{best} ({score}/{len(idx)})' if best else 'mixed'
print('Group identification (which market space):')
for g,idx in groups.items():
    print(f'  group {g}: {label_group(idx)}')

# plot the onset curve
ks=['0','1','2','3+']; Ps=[np.mean(buckets[k]) if buckets[k] else np.nan for k in ks]
fig,ax=plt.subplots(1,2,figsize=(13,5)); fig.suptitle('Onset: do early movers precede a group event?',fontweight='bold')
ax[0].bar(ks,Ps,color='seagreen'); ax[0].axhline(base,color='crimson',ls='--',label=f'base rate {base:.2f}')
ax[0].set_xlabel('# early movers today (group not yet full)'); ax[0].set_ylabel(f'P(full activation within {H}d)')
ax[0].set_title('Onset probability vs early movers'); ax[0].legend(fontsize=9)
# example group frac-active trace
g0=list(groups)[0]
ax[1].plot(frac_act[g0],lw=0.7); ax[1].axhline(ACT_FRAC,color='crimson',ls=':',label=f'full={ACT_FRAC:.0%}')
ax[1].set_title(f'group {g0} fraction-active over time'); ax[1].legend(fontsize=8); ax[1].set_xlabel('snapshot')
plt.tight_layout(); plt.savefig(os.path.join(PLOTS,'onset_detection.png'),dpi=150,bbox_inches='tight'); plt.show()
"""))

cells.append(code("cell-05","""# Cell 5 - Verdict
p1=np.mean(buckets['1']) if buckets['1'] else np.nan
p3=np.mean(buckets['3+']) if buckets['3+'] else np.nan
lift1=p1/base if base>0 else np.nan; lift3=p3/base if base>0 else np.nan
print('VERDICT'); print('='*60)
print(f'Base rate: {base:.3f} | P|1 mover: {p1:.3f} ({lift1:.2f}x) | P|3+ movers: {p3:.3f} ({lift3:.2f}x)')
print(f'Continuous precursor partial corr (vol-controlled): {part:+.3f}')
print()
if (lift3>1.3 and p3>p1>base) or part>0.15:
    print('-> ONSET SIGNAL EXISTS: partial activation precedes group-wide moves')
    print('   beyond volatility clustering. Build the early-warning alert on this.')
elif lift3>1.1 or part>0.08:
    print('-> WEAK onset signal. Some precursor value, but modest -- usable as a')
    print('   soft alert at best; combine with same-day detection.')
else:
    print('-> NO usable onset lead at daily resolution (expected for large-caps that')
    print('   move near-simultaneously). Fall back to SAME-DAY detection +')
    print('   identification, which work and still power the app.')
"""))

nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.11.9"}},"nbformat":4,"nbformat_minor":5}
open("notebooks/onset_detection.ipynb","w").write(json.dumps(nb,indent=1))
print("Wrote notebooks/onset_detection.ipynb with",len(cells),"cells")
