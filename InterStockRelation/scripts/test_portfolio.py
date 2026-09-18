import numpy as np, pickle, os, pandas as pd
DATA='data/processed'; RAW='data/raw'; PREDS='outputs/predictions'
corr=np.load(os.path.join(DATA,'corr_video.npy'))
pm=pickle.load(open(os.path.join(DATA,'pair_meta.pkl'),'rb'))
vm=pickle.load(open(os.path.join(DATA,'video_meta.pkl'),'rb'))
ts=pm['timestamps']; snap_ts=np.asarray(vm['snap_timesteps']); ordered=vm['ordered_tickers']
N_SNAPS,N=corr.shape[0],corr.shape[1]; LOOKBACK=20; HORIZON=5; TRAIN_RATIO=0.70

# returns (ordered) + trailing vols
close=pd.DataFrame(index=ts)
for tk in ordered:
    d=pd.read_csv(os.path.join(RAW,f'{tk}_5minute.csv'),parse_dates=['date']).set_index('date')
    close[tk]=d['close'].reindex(ts)
sc=close.iloc[snap_ts].to_numpy()
ret=np.vstack([np.full((1,N),np.nan),np.log(sc[1:]/sc[:-1])])
vol=np.vstack([np.full((1,N),np.nan), pd.DataFrame(ret[1:]).rolling(LOOKBACK).std().to_numpy()])

# val split + preds
n_valid=N_SNAPS-HORIZON
fm=np.array([np.mean((corr[t]-corr[t+HORIZON])**2) for t in range(n_valid)],dtype=np.float32)
vt=float(np.median(fm)); vs=np.array([s for s in range(n_valid) if s>=LOOKBACK]); sv=fm[vs]
vo=vs[sv>=vt]; st=vs[sv<vt]; np.random.seed(42)
a=np.random.choice(vo,int(len(vo)*TRAIN_RATIO),replace=False); va=np.setdiff1d(vo,a)
c=np.random.choice(st,int(len(st)*TRAIN_RATIO),replace=False); vb=np.setdiff1d(st,c)
val_snaps=np.sort(np.concatenate([va,vb]))
preds=np.load(os.path.join(PREDS,'trial7_return_val_preds.npy'))
mean_corr=corr[np.sort(np.concatenate([a,c]))].mean(0)   # train-mean correlation

def minvar_w(R, sig, delta=0.10):
    R=np.clip((R+R.T)/2,-0.999,0.999); np.fill_diagonal(R,1.0)
    R=(1-delta)*R+delta*np.eye(N)                 # Ledoit-style shrink to identity
    S=np.outer(sig,sig)*R
    try: inv=np.linalg.solve(S,np.ones(N))
    except: inv=np.linalg.pinv(S)@np.ones(N)
    return inv/inv.sum()

estimators={
 'equal-weight': None,
 'persistence (corr[s])': lambda k,s: corr[s],
 'historical mean': lambda k,s: mean_corr,
 'forecast (T7)': lambda k,s: preds[k],
}
realized={name:[] for name in estimators}
for k,s in enumerate(val_snaps):
    sig=vol[s]
    if np.isnan(sig).any() or (sig<1e-9).any(): continue
    fut=ret[s+1:s+HORIZON+1]
    if np.isnan(fut).any(): continue
    for name,fn in estimators.items():
        w=np.ones(N)/N if fn is None else minvar_w(fn(k,s), sig)
        rp=fut@w                                   # portfolio daily returns over holding window
        realized[name].append(rp)
print("Out-of-sample MINIMUM-VARIANCE portfolio realized volatility (lower=better):")
print(f"{'covariance estimator':<24}{'realized vol':>14}{'vs equal-wt':>12}")
print('-'*50)
ew=np.std(np.concatenate(realized['equal-weight']))
for name in estimators:
    rv=np.std(np.concatenate(realized[name]))
    print(f"{name:<24}{rv:>14.5f}{(rv/ew-1)*100:>+11.1f}%")
print("\nThe covariance forecast wins if 'forecast' has the LOWEST realized vol,")
print("especially below persistence (the standard 'use current correlations').")
