"""Build intraday_onset.ipynb - re-run the onset/early-warning test at intraday
resolution (the daily test failed as 'too fast to catch at daily resolution').

Per cadence (5/15/30 min) and per look-ahead window, measure:
  P(group fully activates within next H bars | k early movers now, group not full now)
vs base rate, plus a volatility-controlled continuous precursor. Returns are
computed WITHIN a day only (no overnight jumps). Groups come from KMeans on the
5-min return correlation (ARI 0.75 vs sectors in the feasibility probe).
No training.
"""
import json
def code(cid,src): return {"cell_type":"code","id":cid,"metadata":{},"execution_count":None,"outputs":[],"source":src.splitlines(keepends=True)}
def md(cid,src): return {"cell_type":"markdown","id":cid,"metadata":{},"source":src.splitlines(keepends=True)}
cells=[]
cells.append(md("md-01","""# Intraday Onset Re-test

The daily onset test found *no* early-warning signal (P(full activation) fell as
early movers rose). The explanation was resolution: large-caps move together
within a day, so daily bars can't see who moved first. The feasibility probe
showed intraday correlation structure is stable and sector-coherent. Here we
re-run the same onset test at 5/15/30-minute cadence.

**Onset signal exists** if P(group becomes fully active soon) rises with the
number of early movers *while the group is not yet fully active*, above the
base rate and beyond volatility clustering."""))

cells.append(code("cell-01","""# Cell 1 - Load 5-min closes, day boundaries, intraday groups
import os, pickle, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
_cwd=os.getcwd(); ROOT=os.path.abspath(os.path.join(_cwd,'..')) if os.path.basename(_cwd)=='notebooks' else _cwd
DATA=os.path.join(ROOT,'data','processed'); RAW=os.path.join(ROOT,'data','raw'); PLOTS=os.path.join(ROOT,'outputs','plots')
pm=pickle.load(open(os.path.join(DATA,'pair_meta.pkl'),'rb')); vm=pickle.load(open(os.path.join(DATA,'video_meta.pkl'),'rb'))
ts=pm['timestamps']; ordered=vm['ordered_tickers']; N=len(ordered)
close=pd.DataFrame(index=ts)
for tk in ordered:
    d=pd.read_csv(os.path.join(RAW,f'{tk}_5minute.csv'),parse_dates=['date']).set_index('date')
    close[tk]=d['close'].reindex(ts)
logP=np.log(close.ffill().to_numpy())
day=pd.Series(ts).dt.normalize().to_numpy()

Z_THRESH=2.0; ACT_FRAC=0.40; K=8; W_STD=60
CADENCES={'5min':1,'15min':3,'30min':6}
LOOKAHEAD_MIN=[30,60,120]          # look-ahead windows in minutes

# groups from 5-min returns over the first 60% of the sample (no look-ahead)
r5=np.diff(logP,axis=0); same_day=day[1:]==day[:-1]; r5[~same_day]=np.nan
cut=int(0.6*len(r5)); R=r5[:cut]; R=R[~np.isnan(R).any(1)]
labels=KMeans(n_clusters=K,n_init=10,random_state=42).fit_predict(np.corrcoef(R,rowvar=False))
groups={g:np.where(labels==g)[0] for g in np.unique(labels)}
groups={g:i for g,i in groups.items() if len(i)>=3}
print(f'{len(ts)} bars | {len(groups)} groups from 5-min correlation (train 60%)')
for g,i in groups.items(): print(f'  g{g} n={len(i)}: {[ordered[j] for j in i][:6]}')
"""))

cells.append(code("cell-02","""# Cell 2 - Onset test per cadence and look-ahead (evaluated on the last 40%)
def cadence_returns(k):
    # non-overlapping k-bar returns within each day; returns (T_k, N) and bar index
    idx=[]; rets=[]
    bounds=np.r_[0,np.where(day[1:]!=day[:-1])[0]+1,len(day)]
    for s,e in zip(bounds[:-1],bounds[1:]):
        lp=logP[s:e]
        if len(lp)<=k: continue
        r=lp[k::k]-lp[:-k:k][:len(lp[k::k])]
        rets.append(r); idx.extend(range(s+k,s+k*(len(r)+1),k)[:len(r)])
        idx.append(-1)            # marker: day break
        rets.append(np.full((1,N),np.nan))
    return np.vstack(rets), np.array(idx)

results=[]
for cname,k in CADENCES.items():
    ret,bidx=cadence_returns(k)
    std=pd.DataFrame(ret).rolling(W_STD,min_periods=20).std().shift(1).to_numpy()
    z=ret/np.where(std>1e-12,std,np.nan); act=np.abs(z)>Z_THRESH
    T=len(ret); start=int(0.6*T)
    for la in LOOKAHEAD_MIN:
        H=max(1,la//(5*k))
        buckets={'0':[],'1':[],'2':[],'3+':[]}; X=[]; Yc=[]; Yf=[]
        for g,ix in groups.items():
            fa=np.nanmean(act[:,ix],axis=1); kc=np.nansum(act[:,ix],axis=1)
            full=fa>=ACT_FRAC; inten=np.nanmean(np.abs(z[:,ix]),axis=1)
            for t in range(start,T-H):
                if np.isnan(ret[t]).all() or full[t]: continue
                win=ret[t+1:t+1+H]
                if np.isnan(win).all(axis=1).any(): continue   # don't look across a day break
                ff=bool(full[t+1:t+1+H].any())
                kk=int(kc[t]); key='0' if kk==0 else '1' if kk==1 else '2' if kk==2 else '3+'
                buckets[key].append(ff)
                X.append(fa[t]); Yc.append(inten[t]); Yf.append(np.nanmean(inten[t+1:t+1+H]))
        allv=sum(buckets.values(),[]); base=np.mean(allv) if allv else np.nan
        X,Yc,Yf=map(np.array,(X,Yc,Yf)); m=~(np.isnan(X)|np.isnan(Yc)|np.isnan(Yf))
        b=np.polyfit(Yc[m],Yf[m],1); part=pearsonr(X[m],Yf[m]-np.polyval(b,Yc[m]))[0]
        row=dict(cadence=cname,lookahead=f'{la}m',base=base,partial=part,n=len(allv))
        for key in buckets: row[f'P|k={key}']=np.mean(buckets[key]) if buckets[key] else np.nan
        results.append(row)
        print(f\"{cname:>6} +{la:>3}m | base {base:.3f} | k0 {row['P|k=0']:.3f} k1 {row['P|k=1']:.3f} \"
              f\"k2 {row['P|k=2']:.3f} k3+ {row['P|k=3+']:.3f} | vol-controlled partial {part:+.3f}\")
df=pd.DataFrame(results)
"""))

cells.append(code("cell-03","""# Cell 3 - Plot + verdict
fig,ax=plt.subplots(1,len(CADENCES),figsize=(15,4.5),sharey=True)
fig.suptitle('Intraday onset: P(group fully activates soon) vs early movers',fontweight='bold')
for a,c in zip(ax,CADENCES):
    sub=df[df.cadence==c]
    for _,r in sub.iterrows():
        a.plot(['0','1','2','3+'],[r[f'P|k={x}'] for x in ['0','1','2','3+']],'o-',label=f\"+{r['lookahead']}\")
    a.set_title(c); a.set_xlabel('early movers now'); a.grid(alpha=0.3); a.legend(fontsize=8)
ax[0].set_ylabel('P(full activation)')
plt.tight_layout(); plt.savefig(os.path.join(PLOTS,'intraday_onset.png'),dpi=150,bbox_inches='tight'); plt.show()

df['lift3']=df['P|k=3+']/df['base']; df['mono']=(df['P|k=1']>df['P|k=0'])&(df['P|k=3+']>df['P|k=1'])
print(df[['cadence','lookahead','base','P|k=0','P|k=1','P|k=3+','lift3','partial','mono']].to_string(index=False,float_format='%.3f'))
best=df.loc[df['lift3'].idxmax()]
print('\\nVERDICT'); print('='*60)
if (df['mono']&(df['lift3']>1.3)).any() or (df['partial']>0.10).any():
    print(f\"-> ONSET SIGNAL at intraday resolution (best: {best.cadence} +{best.lookahead}, lift {best.lift3:.2f}x).\")
    print('   Daily resolution was hiding it. Worth building the early-warning alert on this.')
elif (df['lift3']>1.1).any():
    print(f'-> WEAK intraday onset signal (best lift {best.lift3:.2f}x). Marginal.')
else:
    print('-> No onset lead even intraday. Large-caps move together within minutes;')
    print('   the onset hypothesis is closed. Fall back to same-bar detection + identification.')
"""))
cells.append(md("md-04","""## Volatility-stratified check

The lift above could be volatility clustering: a volatile moment has more movers
*and* is more likely to become fully active, whether or not the movers lead.
Here we hold current volatility fixed. Within each quintile of current group
intensity (mean |z|, plus recent past intensity), does 3+ movers still beat 0
movers? We also fit a logistic model with the volatility controls and check the
mover coefficient, and repeat per group."""))

cells.append(code("cell-04","""# Cell 4 - Lift within volatility strata, logistic control, per-group
from sklearn.linear_model import LogisticRegression
def obs_table(k,H):
    ret,_=cadence_returns(k)
    std=pd.DataFrame(ret).rolling(W_STD,min_periods=20).std().shift(1).to_numpy()
    z=ret/np.where(std>1e-12,std,np.nan); act=np.abs(z)>Z_THRESH
    T=len(ret); start=int(0.6*T); brk=np.isnan(ret).all(1)
    rows=[]
    for g,ix in groups.items():
        fa=np.nanmean(act[:,ix],1); kc=np.nansum(act[:,ix],1); full=fa>=ACT_FRAC
        inten=pd.Series(np.nanmean(np.abs(z[:,ix]),1))
        past=inten.shift(1).rolling(6,min_periods=1).mean().to_numpy()   # recent vol regime
        fut=pd.Series(full.astype(float)).shift(-1).rolling(H).max().shift(-(H-1)).to_numpy()
        badw=pd.Series(brk.astype(float)).shift(-1).rolling(H).max().shift(-(H-1)).to_numpy()
        t=np.arange(start,T-H)
        ok=(~brk[t])&(~full[t])&(badw[t]==0)&~np.isnan(inten.to_numpy()[t])&~np.isnan(past[t])
        t=t[ok]
        rows.append(pd.DataFrame(dict(g=g,k=kc[t],kf=kc[t]/len(ix),inten=inten.to_numpy()[t],past=past[t],y=fut[t])))
    d=pd.concat(rows,ignore_index=True)
    d['kb']=np.select([d.k==0,d.k==1,d.k==2],['0','1','2'],'3+')
    d['q']=d.groupby('g')['inten'].transform(lambda s: pd.qcut(s.rank(method='first'),5,labels=False))
    return d

strat=[]
for cname,k in [('5min',1),('15min',3)]:
    for la in [30,60]:
        H=max(1,la//(5*k)); d=obs_table(k,H)
        print(f'\\n=== {cname} +{la}m  (n={len(d)}) ===')
        print('vol quintile | P|k=0  P|k=1  P|k=3+ | lift(3+/0)   n(3+)')
        lifts=[]
        for q in range(5):
            s=d[d.q==q]; p=s.groupby('kb')['y'].mean(); n3=(s.kb=='3+').sum()
            l=p.get('3+',np.nan)/p.get('0',np.nan) if n3>=30 else np.nan; lifts.append(l)
            print(f'     Q{q+1}      | {p.get(\"0\",np.nan):.3f}  {p.get(\"1\",np.nan):.3f}  {p.get(\"3+\",np.nan):.3f}  |   {l:.2f}x     {n3}')
        X=d[['kf','inten','past']].to_numpy(); X=(X-X.mean(0))/X.std(0)
        lr=LogisticRegression().fit(X,d.y); b_kf=lr.coef_[0][0]
        lr0=LogisticRegression().fit(X[:,1:],d.y)
        from sklearn.metrics import roc_auc_score
        auc1=roc_auc_score(d.y,lr.predict_proba(X)[:,1]); auc0=roc_auc_score(d.y,lr0.predict_proba(X[:,1:])[:,1])
        print(f'logit: mover-fraction coef {b_kf:+.3f} (std units) | AUC vol-only {auc0:.3f} -> +movers {auc1:.3f}')
        pg=[]
        for g,s in d.groupby('g'):
            p=s.groupby('kb')['y'].mean()
            if (s.kb=='3+').sum()>=30 and p.get('0',0)>0: pg.append((g,p['3+']/p['0']))
        print('per-group lift(3+/0) controlling nothing: '+', '.join(f'g{g}:{l:.2f}' for g,l in pg))
        strat.append(dict(cadence=cname,lookahead=f'{la}m',med_strat_lift=np.nanmedian(lifts),
                          coef=b_kf,dAUC=auc1-auc0))

S=pd.DataFrame(strat); print('\\n'+S.to_string(index=False,float_format='%.3f'))
print('\\nVERDICT (volatility-controlled)'); print('='*60)
if (S.med_strat_lift>1.2).any() and (S.dAUC>0.01).any():
    print('-> Onset survives volatility control: early movers carry lead information')
    print('   beyond current volatility. Real early-warning candidate.')
elif (S.med_strat_lift>1.05).any():
    print('-> Mostly volatility clustering; a small residual lead remains. Marginal for alerts.')
else:
    print('-> Lift is volatility clustering. No genuine onset lead.')
"""))
nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.11.9"}},"nbformat":4,"nbformat_minor":5}
open("notebooks/intraday_onset.ipynb","w").write(json.dumps(nb,indent=1))
print("Wrote notebooks/intraday_onset.ipynb with",len(cells),"cells")
