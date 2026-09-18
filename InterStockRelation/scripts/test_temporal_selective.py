import numpy as np, os
DATA='data/processed'; PREDS='outputs/predictions'
corr=np.load(os.path.join(DATA,'corr_video.npy'))
N_SNAPS=corr.shape[0]; N=95; LOOKBACK=20; HORIZON=5; TRAIN_RATIO=0.70
# rebuild val split (seed 42) - val_snaps are time-sorted
n_valid=N_SNAPS-HORIZON
fm=np.array([np.mean((corr[t]-corr[t+HORIZON])**2) for t in range(n_valid)],dtype=np.float32)
vt=float(np.median(fm)); vs=np.array([s for s in range(n_valid) if s>=LOOKBACK]); sv=fm[vs]
vo=vs[sv>=vt]; st=vs[sv<vt]; np.random.seed(42)
a=np.random.choice(vo,int(len(vo)*TRAIN_RATIO),replace=False); va=np.setdiff1d(vo,a)
c=np.random.choice(st,int(len(st)*TRAIN_RATIO),replace=False); vb=np.setdiff1d(st,c)
val_snaps=np.sort(np.concatenate([va,vb]))
preds=np.load(os.path.join(PREDS,'trial7_return_val_preds.npy'))
iu,ju=np.triu_indices(N,k=1)

def cov_curve(mask_idx):
    # mask_idx: boolean over val rows to include
    cur=corr[val_snaps-1][mask_idx]; fut=corr[val_snaps+HORIZON][mask_idx]; pr=preds[mask_idx]
    tc=(fut-cur)[:,iu,ju].reshape(-1); pc=(pr-cur)[:,iu,ju].reshape(-1)
    t=np.sign(tc); pm=np.sign(pc); conf=np.abs(pc); nz=t!=0
    order=np.argsort(conf)[::-1]; t_o,pm_o=t[order],pm[order]; nzo=t_o!=0
    res={}
    for q in [0.10,0.25,0.50,1.00]:
        k=max(1,int(q*len(order))); sel=nzo[:k]
        res[q]=float((pm_o[:k][sel]==t_o[:k][sel]).mean())
    return res

nval=len(val_snaps); half=nval//2
first=np.zeros(nval,bool); first[:half]=True   # earlier dates (val_snaps sorted)
last=~first
recent=np.zeros(nval,bool); recent[int(nval*0.8):]=True  # last 20% by date

print("Selective-prediction dir-acc by coverage, across TIME partitions:")
print(f"{'partition':<16}{'@10%':>8}{'@25%':>8}{'@50%':>8}{'@100%':>8}")
print('-'*48)
for label,mask in [('all val',np.ones(nval,bool)),('first half',first),('second half',last),('last 20%',recent)]:
    r=cov_curve(mask)
    print(f"{label:<16}{r[0.10]:>8.3f}{r[0.25]:>8.3f}{r[0.50]:>8.3f}{r[1.00]:>8.3f}")
print("\nRobust if the curve holds across first/second half and the recent period.")
