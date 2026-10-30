"""Build intraday_probe.ipynb - feasibility probe BEFORE committing to an
intraday pipeline.

Three questions, all cheap (no training, no full video build):
  (a) EPPS EFFECT: how much does measured correlation attenuate as sampling
      frequency rises (5min -> 15 -> 30 -> 60 -> daily)? Quantify it in OUR data.
  (b) STRUCTURE SURVIVAL: does sector/block structure persist intraday? Epps
      shifts the LEVEL of all correlations; what matters is whether the RELATIVE
      structure (within-sector vs cross-sector contrast) survives.
  (c) STABILITY: how noisy is the estimate at each cadence (similarity of
      adjacent-window estimates)?

Decision this informs: which cadence to build the intraday correlation video at
(or whether to switch to an asynchronous-aware estimator instead).
"""
import json
def code(cid,src): return {"cell_type":"code","id":cid,"metadata":{},"execution_count":None,"outputs":[],"source":src.splitlines(keepends=True)}
def md(cid,src): return {"cell_type":"markdown","id":cid,"metadata":{},"source":src.splitlines(keepends=True)}

cells=[]
cells.append(md("md-01","""# Intraday Feasibility Probe

Everything in this project so far is **daily** (1,157 snapshots, stride 75 bars).
We hold 87,122 five-minute bars and discarded that resolution in Trial 2. The
onset test failed with the explicit explanation *"too fast to catch at daily
resolution"* --- so intraday is the one untested axis with unused data.

Before building an intraday pipeline, this probe answers:

1. **Epps effect** --- measured correlations decay toward zero at higher
   sampling frequency (asynchronous trading). How bad is it in our data?
2. **Structure survival** --- Epps shifts the *level* of all correlations. The
   real question is whether the *relative* structure (sector blocks) survives.
3. **Stability** --- how noisy is the correlation estimate at each cadence?

Output: a recommended cadence, or a verdict that we need an
asynchronous-aware estimator (Hayashi-Yoshida) instead of plain Pearson."""))

cells.append(code("cell-01","""# Cell 1 - Load 5-min closes on the processed timeline + sector map
import os, pickle, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib.pyplot as plt

_cwd=os.getcwd()
ROOT=os.path.abspath(os.path.join(_cwd,'..')) if os.path.basename(_cwd)=='notebooks' else _cwd
DATA=os.path.join(ROOT,'data','processed'); RAW=os.path.join(ROOT,'data','raw'); PLOTS=os.path.join(ROOT,'outputs','plots')
pm=pickle.load(open(os.path.join(DATA,'pair_meta.pkl'),'rb'))
vm=pickle.load(open(os.path.join(DATA,'video_meta.pkl'),'rb'))
ts=pm['timestamps']; ordered=vm['ordered_tickers']; N=len(ordered)

close=pd.DataFrame(index=ts)
for tk in ordered:
    d=pd.read_csv(os.path.join(RAW,f'{tk}_5minute.csv'),parse_dates=['date']).set_index('date')
    close[tk]=d['close'].reindex(ts)
logP=np.log(close.to_numpy())
day_id=pd.Series(ts).dt.normalize().to_numpy()
udays,day_start=np.unique(day_id,return_index=True)
day_start=np.sort(day_start); day_bounds=list(day_start)+[len(ts)]
print(f'{N} stocks | {len(ts)} five-min bars | {len(udays)} trading days')

SECTORS={
 'BANK':['HDFCBANK','ICICIBANK','SBIN','AXISBANK','KOTAKBANK','BANKBARODA','PNB','CANBK','UNIONBANK'],
 'IT':['TCS','INFY','WIPRO','HCLTECH','TECHM','LTM'],
 'PHARMA':['SUNPHARMA','CIPLA','DRREDDY','DIVISLAB','TORNTPHARM','ZYDUSLIFE'],
 'ENERGY':['RELIANCE','ONGC','BPCL','IOC','GAIL','NTPC','POWERGRID','COALINDIA','TATAPOWER'],
 'METAL':['TATASTEEL','JSWSTEEL','JINDALSTEL','HINDALCO','VEDL','HINDZINC'],
 'AUTO':['MARUTI','MM','TVSMOTOR','EICHERMOT','BAJAJ-AUTO','BOSCHLTD','MOTHERSON'],
 'FMCG':['HINDUNILVR','ITC','NESTLEIND','BRITANNIA','GODREJCP','TATACONSUM'],
}
pos={tk:i for i,tk in enumerate(ordered)}
sec_of=np.full(N,-1)
for si,(s,mem) in enumerate(SECTORS.items()):
    for tk in mem:
        if tk in pos: sec_of[pos[tk]]=si
print('sector-tagged stocks:',int((sec_of>=0).sum()),'of',N)
"""))

cells.append(code("cell-02","""# Cell 2 - k-bar non-overlapping returns (never spanning overnight) + corr matrix
BARS_PER_DAY=75
KS={'5min':1,'15min':3,'30min':6,'60min':12,'daily':BARS_PER_DAY}

def returns_k(b0,b1,k):
    \"\"\"Non-overlapping k-bar log returns in [b0,b1).
    Intraday (k < BARS_PER_DAY): never crosses a day boundary.
    Daily (k >= BARS_PER_DAY): close-to-close across days (a day holds only
    ~75 bars, so a 75-bar return cannot live inside one day).\"\"\"
    if k>=BARS_PER_DAY:
        closes=[]
        for ds,de in zip(day_bounds[:-1],day_bounds[1:]):
            s,e=max(ds,b0),min(de,b1)
            if e<=s: continue
            closes.append(logP[e-1])          # last bar of each day
        if len(closes)<2: return np.empty((0,N))
        A=np.vstack(closes)
        return A[1:]-A[:-1]
    out=[]
    for ds,de in zip(day_bounds[:-1],day_bounds[1:]):
        s,e=max(ds,b0),min(de,b1)
        if e-s<=k: continue
        lp=logP[s:e]
        r=lp[k::k]-lp[:-k:k][:len(lp[k::k])]
        out.append(r)
    return np.vstack(out) if out else np.empty((0,N))

def corr_of(b0,b1,k,min_obs=30):
    R=returns_k(b0,b1,k)
    R=R[~np.isnan(R).any(1)]
    if len(R)<min_obs: return None,len(R)
    C=np.corrcoef(R,rowvar=False)
    return C,len(R)

iu,ju=np.triu_indices(N,k=1)
both=(sec_of[iu]>=0)&(sec_of[ju]>=0)
same=both&(sec_of[iu]==sec_of[ju])
diff=both&(sec_of[iu]!=sec_of[ju])
print('pairs: same-sector',int(same.sum()),'| cross-sector',int(diff.sum()))
"""))

cells.append(code("cell-03","""# Cell 3 - (a) EPPS + (b) STRUCTURE across cadences, over several spans
SPAN_DAYS=60
spans=[]
for i in range(0,len(day_bounds)-1-SPAN_DAYS,SPAN_DAYS):
    spans.append((day_bounds[i],day_bounds[i+SPAN_DAYS]))
spans=spans[:10]
print(f'{len(spans)} non-overlapping spans of {SPAN_DAYS} trading days\\n')

rows=[]
mats={name:[] for name in KS}
for name,k in KS.items():
    mc=[]; wi=[]; cr=[]; ev=[]; nobs=[]
    for (b0,b1) in spans:
        C,n=corr_of(b0,b1,k)
        if C is None: continue
        off=C[iu,ju]
        mc.append(np.nanmean(off)); wi.append(np.nanmean(off[same])); cr.append(np.nanmean(off[diff]))
        ev.append(np.linalg.eigvalsh(np.nan_to_num(C,nan=0.0))[-1]/N); nobs.append(n)
        mats[name].append(C)
    if not nobs:
        print(f'  [skip] {name}: no span produced enough observations'); continue
    contrast=np.mean(wi)-np.mean(cr)
    rows.append(dict(cadence=name, obs=int(np.mean(nobs)), mean_corr=np.mean(mc),
                     within=np.mean(wi), cross=np.mean(cr), contrast=contrast,
                     norm_contrast=contrast/max(1e-9,np.mean(mc)), lam1_frac=np.mean(ev)))
df=pd.DataFrame(rows)
print('EPPS + STRUCTURE by sampling cadence (averaged over spans):')
print(f'{\"cadence\":<9}{\"obs\":>7}{\"mean_corr\":>11}{\"within\":>9}{\"cross\":>8}{\"contrast\":>10}{\"norm_con\":>10}{\"lam1/N\":>8}')
print('-'*72)
for r in rows:
    print(f'{r[\"cadence\"]:<9}{r[\"obs\"]:>7}{r[\"mean_corr\"]:>11.3f}{r[\"within\"]:>9.3f}{r[\"cross\"]:>8.3f}{r[\"contrast\"]:>10.3f}{r[\"norm_contrast\"]:>10.3f}{r[\"lam1_frac\"]:>8.3f}')
print('\\nEPPS: mean_corr should RISE from 5min -> daily (attenuation at high freq).')
print('STRUCTURE SURVIVES if contrast (within-cross) stays clearly positive, and')
print('norm_contrast (contrast relative to the level) stays comparable to daily.')
"""))

cells.append(code("cell-04","""# Cell 4 - (c) STABILITY: similarity of adjacent-window estimates per cadence
print('Stability: correlation between adjacent spans\\' off-diagonal estimates')
print(f'{\"cadence\":<9}{\"adj-span similarity\":>22}')
print('-'*32)
stab={}
for name in KS:
    Ms=mats[name]
    if len(Ms)<2: continue
    sims=[]
    for A,B in zip(Ms[:-1],Ms[1:]):
        a,b=A[iu,ju],B[iu,ju]; m=~(np.isnan(a)|np.isnan(b))
        sims.append(np.corrcoef(a[m],b[m])[0,1])
    stab[name]=np.mean(sims)
    print(f'{name:<9}{np.mean(sims):>22.3f}')
print('\\n(higher = the structure estimate is reproducible across time windows;')
print(' very low at a cadence means the estimate is dominated by noise)')

# Can KMeans still recover sectors intraday?
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
print('\\nSector recovery from the correlation matrix (ARI vs true sector tags):')
tagged=sec_of>=0
for name in KS:
    if not mats[name]: continue
    C=np.nan_to_num(np.mean(mats[name],axis=0),nan=0.0)
    lab=KMeans(n_clusters=len(SECTORS),n_init=10,random_state=42).fit_predict(C)
    print(f'  {name:<8} ARI={adjusted_rand_score(sec_of[tagged],lab[tagged]):.3f}')
"""))

cells.append(code("cell-05","""# Cell 5 - Plot + verdict
fig,ax=plt.subplots(1,3,figsize=(16,4.5)); fig.suptitle('Intraday feasibility probe',fontweight='bold')
names=[r['cadence'] for r in rows]; x=np.arange(len(names))
ax[0].plot(x,[r['mean_corr'] for r in rows],'o-',label='mean corr')
ax[0].plot(x,[r['within'] for r in rows],'s-',label='within-sector')
ax[0].plot(x,[r['cross'] for r in rows],'^-',label='cross-sector')
ax[0].set_xticks(x); ax[0].set_xticklabels(names); ax[0].set_title('Epps: correlation vs cadence'); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
ax[1].bar(x,[r['norm_contrast'] for r in rows],color='seagreen')
ax[1].set_xticks(x); ax[1].set_xticklabels(names); ax[1].set_title('Normalized sector contrast\\n(structure survival)'); ax[1].grid(alpha=0.3)
ax[2].bar(x,[stab.get(n,np.nan) for n in names],color='steelblue')
ax[2].set_xticks(x); ax[2].set_xticklabels(names); ax[2].set_title('Estimate stability\\n(adjacent-span similarity)'); ax[2].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(PLOTS,'intraday_probe.png'),dpi=150,bbox_inches='tight'); plt.show()

d=df.set_index('cadence')
epps_drop=(d.loc['daily','mean_corr']-d.loc['5min','mean_corr'])/max(1e-9,d.loc['daily','mean_corr'])
print('VERDICT'); print('='*64)
print(f'Epps attenuation 5min vs daily: {epps_drop*100:.0f}% of the daily correlation level is lost at 5-min')
for cad in ['5min','15min','30min','60min']:
    if cad in d.index:
        keep=d.loc[cad,'norm_contrast']/max(1e-9,d.loc['daily','norm_contrast'])
        print(f'  {cad:<6} normalized sector contrast = {keep*100:.0f}% of daily | stability {stab.get(cad,float(\"nan\")):.2f}')
print()
ok=[c for c in ['5min','15min','30min','60min'] if c in d.index and
    d.loc[c,'norm_contrast']/max(1e-9,d.loc['daily','norm_contrast'])>0.6 and stab.get(c,0)>0.5]
if ok:
    print(f'-> Structure SURVIVES at: {\", \".join(ok)}. Build the intraday video at the')
    print(f'   finest of these ({ok[0]}) and re-run the onset test there.')
else:
    print('-> Structure does NOT survive cleanly at any intraday cadence with plain')
    print('   Pearson. Either go coarser than 60min, or switch to an asynchronous-aware')
    print('   estimator (Hayashi-Yoshida) before building the pipeline.')
"""))

nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.11.9"}},"nbformat":4,"nbformat_minor":5}
open("notebooks/intraday_probe.ipynb","w").write(json.dumps(nb,indent=1))
print("Wrote notebooks/intraday_probe.ipynb with",len(cells),"cells")
