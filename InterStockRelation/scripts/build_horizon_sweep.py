"""Build horizon_sweep.ipynb - train the ConvLSTM at H = 1, 2, 3, 5 and compare.

The no-training ablation suggested predictable signal concentrates at the shortest
horizon (frame-delta autocorrelation ~ -0.47 at H=1, ~0 by H>=2). This is the one
predictive lever we never actually TRAINED. Here we retrain the return-only
5-channel ConvLSTM at each horizon and compare aggregate metrics AND the
selective-prediction coverage curve. If H=1 lifts the confident-call ceiling, the
app should issue short-horizon alerts.

~7 min per horizon on the RTX 4070 -> ~30 min total. Run top to bottom.
"""
import json
def code(cid,src): return {"cell_type":"code","id":cid,"metadata":{},"execution_count":None,"outputs":[],"source":src.splitlines(keepends=True)}
def md(cid,src): return {"cell_type":"markdown","id":cid,"metadata":{},"source":src.splitlines(keepends=True)}
cells=[]
cells.append(md("md-01","""# Horizon Sweep: Train at H = 1, 2, 3, 5

Tests whether shorter horizons lift the predictability / confident-call ceiling.
Same return-only 5-channel ConvLSTM, combined loss, cosine schedule; only the
prediction horizon H changes. Reports per-horizon MSE / Pearson / dir-acc and the
selective-prediction coverage curve (dir-acc at 10/25/50/100% coverage)."""))

cells.append(code("cell-01","""# Cell 1 - Config + data + 5-channel video
import os, time, pickle, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import pearsonr
from tqdm import tqdm
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
_cwd=os.getcwd(); ROOT=os.path.abspath(os.path.join(_cwd,'..')) if os.path.basename(_cwd)=='notebooks' else _cwd
DATA=os.path.join(ROOT,'data','processed'); MODELS=os.path.join(ROOT,'models'); PREDS=os.path.join(ROOT,'outputs','predictions'); PLOTS=os.path.join(ROOT,'outputs','plots')
N_STOCKS=95; LOOKBACK=20; TRAIN_RATIO=0.70; BATCH=16; LR=1e-3; MAX_EPOCHS=60; WD=1e-4
T_0,T_MULT,ETA_MIN,PATIENCE=20,2,1e-6,15
ALPHA,BETA,GAMMA,DELTA,DIRTH=0.35,0.25,0.30,0.10,0.005
HORIZONS=[1,2,3,5]
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); PIN=device.type=='cuda'
corr=np.load(os.path.join(DATA,'corr_video.npy')); N_SNAPS=corr.shape[0]
def build_sub(v,ds=1,dl=5,win=4):
    T,H,W=v.shape; o=np.zeros((T,5,H,W),np.float32); o[:,0]=v
    o[ds:,1]=v[ds:]-v[:-ds]; o[dl:,2]=v[dl:]-v[:-dl]
    c,c2=np.cumsum(v,0),np.cumsum(v**2,0)
    for t in range(T):
        lo=max(0,t-win+1); n=t-lo+1; s=c[t]-(c[lo-1] if lo>0 else 0.0); s2=c2[t]-(c2[lo-1] if lo>0 else 0.0)
        mn=s/n; o[t,3]=mn; o[t,4]=np.sqrt(np.maximum(0,s2/n-mn**2))
    return o
SUB=build_sub(corr)
print('device',device,'| horizons',HORIZONS)
"""))

cells.append(code("cell-02","""# Cell 2 - Dataset / model / loss / train (H-parametrized)
class DS(Dataset):
    def __init__(s,snaps,H): s.snaps=snaps; s.H=H
    def __len__(s): return len(s.snaps)
    def __getitem__(s,i):
        t=s.snaps[i]
        return (torch.from_numpy(SUB[t-LOOKBACK:t].astype(np.float32)),
                torch.from_numpy(corr[t+s.H].astype(np.float32)),
                torch.from_numpy(corr[t-1].astype(np.float32)),
                torch.tensor(float(frame_mse[t])))
def combined_loss(pred,tgt,cur,w):
    ps=((pred-tgt)**2).mean(dim=(1,2)); ww=w/w.sum()*len(w); mse=(ww*ps).mean()
    pf,tf=pred.flatten(1),tgt.flatten(1); pc=pf-pf.mean(1,keepdim=True); tc=tf-tf.mean(1,keepdim=True)
    pe=(pc*tc).sum(1)/(pc.norm(dim=1)*tc.norm(dim=1)+1e-8); pl=(1-pe).mean()
    pd_,td_=pred-cur,tgt-cur; mask=(td_.abs()>DIRTH).float()
    dl=(F.relu(-(pd_*td_))*mask).sum()/mask.sum().clamp(min=1.0); tl=(pd_.abs()-td_.abs()).pow(2).mean()
    return ALPHA*mse+BETA*pl+GAMMA*dl+DELTA*tl
class Cell(nn.Module):
    def __init__(s,ic,hc,k=3): super().__init__(); s.hc=hc; s.c=nn.Conv2d(ic+hc,4*hc,k,padding=k//2)
    def forward(s,x,h,c): i,f,g,o=s.c(torch.cat([x,h],1)).chunk(4,1); c=torch.sigmoid(f)*c+torch.sigmoid(i)*torch.tanh(g); return torch.sigmoid(o)*torch.tanh(c),c
    def init(s,B,sp,d): z=torch.zeros(B,s.hc,sp,sp,device=d); return z,z.clone()
class ConvLSTM(nn.Module):
    def __init__(s,nch=5,n=95,sp=24,hc=32,dp=0.3):
        super().__init__(); s.n,s.sp,s.hc=n,sp,hc
        s.down=nn.Sequential(nn.Conv2d(nch,16,3,padding=1),nn.BatchNorm2d(16),nn.ReLU(),nn.Conv2d(16,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.AdaptiveAvgPool2d((sp,sp)),nn.Conv2d(32,hc,1))
        s.c1=Cell(hc,hc); s.c2=Cell(hc,hc); s.ln=nn.LayerNorm([hc,sp,sp]); s.dp=nn.Dropout2d(dp)
        f=hc*sp*sp; s.dec=nn.Sequential(nn.Flatten(),nn.Linear(f,1024),nn.ReLU(),nn.Dropout(dp),nn.Linear(1024,2048),nn.ReLU(),nn.Dropout(dp),nn.Linear(2048,n*n),nn.Tanh())
    def forward(s,fr):
        B,L,C,H,W=fr.shape; e=s.down(fr.reshape(B*L,C,H,W)).reshape(B,L,s.hc,s.sp,s.sp)
        h1,c1=s.c1.init(B,s.sp,fr.device); h2,c2=s.c2.init(B,s.sp,fr.device)
        for t in range(L): h1,c1=s.c1(e[:,t],h1,c1); h1=s.dp(h1); h2,c2=s.c2(h1,h2,c2)
        return s.dec(s.ln(h2)).reshape(B,s.n,s.n)
def run(m,dl,opt,tr):
    m.train() if tr else m.eval(); P,Fu=[],[]; ctx=torch.enable_grad() if tr else torch.no_grad()
    with ctx:
        for fr,fu,cu,w in dl:
            fr,fu,cu,w=fr.to(device),fu.to(device),cu.to(device),w.to(device); p=m(fr); l=combined_loss(p,fu,cu,w)
            if tr: opt.zero_grad(); l.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
            P.append(p.detach().cpu().numpy()); Fu.append(fu.cpu().numpy())
    return np.concatenate(P),np.concatenate(Fu)
print('model/loss ready')
"""))

cells.append(code("cell-03","""# Cell 3 - Train at each horizon + selective-prediction coverage
iu,ju=np.triu_indices(N_STOCKS,k=1)
def selective(preds,val_snaps,H):
    cur=corr[val_snaps-1]; fut=corr[val_snaps+H]
    tc=(fut-cur)[:,iu,ju].reshape(-1); pc=(preds-cur)[:,iu,ju].reshape(-1)
    t=np.sign(tc); pm=np.sign(pc); conf=np.abs(pc); order=np.argsort(conf)[::-1]
    t_o,pm_o=t[order],pm[order]; nz=t_o!=0; res={}
    for q in [0.10,0.25,0.50,1.00]:
        k=max(1,int(q*len(order))); sel=nz[:k]; res[q]=float((pm_o[:k][sel]==t_o[:k][sel]).mean())
    return res

results={}
for H in HORIZONS:
    global frame_mse
    n_valid=N_SNAPS-H
    frame_mse=np.array([np.mean((corr[t]-corr[t+H])**2) for t in range(n_valid)],dtype=np.float32)
    thr=float(np.median(frame_mse)); vs=np.array([s for s in range(n_valid) if s>=LOOKBACK]); sv=frame_mse[vs]
    vo=vs[sv>=thr]; stb=vs[sv<thr]; np.random.seed(42)
    a=np.random.choice(vo,int(len(vo)*TRAIN_RATIO),replace=False); va=np.setdiff1d(vo,a)
    c=np.random.choice(stb,int(len(stb)*TRAIN_RATIO),replace=False); vb=np.setdiff1d(stb,c)
    tr_s=np.sort(np.concatenate([a,c])); val_s=np.sort(np.concatenate([va,vb]))
    tdl=DataLoader(DS(tr_s,H),batch_size=BATCH,shuffle=True,pin_memory=PIN)
    vdl=DataLoader(DS(val_s,H),batch_size=BATCH,shuffle=False,pin_memory=PIN)
    m=ConvLSTM().to(device); opt=torch.optim.Adam(m.parameters(),lr=LR,weight_decay=WD)
    sch=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=T_0,T_mult=T_MULT,eta_min=ETA_MIN)
    best=1e9; pat=0; bp=None; vcur=corr[val_s-1]; vfut=corr[val_s+H]
    print(f'\\n=== H={H}  train {len(tr_s)} val {len(val_s)} ===')
    for ep in range(1,MAX_EPOCHS+1):
        run(m,tdl,opt,True); vp,vf=run(m,vdl,opt,False); sch.step()
        mse=float(np.mean((vp-vfut)**2))
        if mse<best: best=mse; pat=0; bp=vp.copy()
        else:
            pat+=1
            if pat>=PATIENCE: break
    da=float(np.mean(np.sign(bp-vcur)==np.sign(vfut-vcur)))
    r,_=pearsonr(bp.reshape(-1)[:500000],vfut.reshape(-1)[:500000])
    sel=selective(bp,val_s,H)
    results[H]=dict(mse=best,dir=da,pearson=r,**{f'cov{int(q*100)}':v for q,v in sel.items()})
    print(f'H={H}: MSE={best:.5f} dir={da:.3f} r={r:.3f} | selective @50%={sel[0.50]:.3f} @25%={sel[0.25]:.3f} @10%={sel[0.10]:.3f}')
print('\\ndone')
"""))

cells.append(code("cell-04","""# Cell 4 - Comparison table + plot + verdict
import pandas as pd
df=pd.DataFrame(results).T
df.index.name='H'
print(df.to_string(float_format='%.4f'))
fig,ax=plt.subplots(1,2,figsize=(14,5)); fig.suptitle('Horizon sweep',fontweight='bold')
Hs=list(results)
ax[0].plot(Hs,[results[h]['dir'] for h in Hs],'o-',label='aggregate dir-acc')
ax[0].plot(Hs,[results[h]['cov50'] for h in Hs],'s-',label='selective @50%')
ax[0].plot(Hs,[results[h]['cov25'] for h in Hs],'^-',label='selective @25%')
ax[0].plot(Hs,[results[h]['cov10'] for h in Hs],'d-',label='selective @10%')
ax[0].set_xlabel('horizon H'); ax[0].set_ylabel('directional accuracy'); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
ax[1].plot(Hs,[results[h]['mse'] for h in Hs],'o-',color='crimson'); ax[1].set_xlabel('horizon H'); ax[1].set_ylabel('val MSE'); ax[1].grid(alpha=0.3); ax[1].set_title('MSE vs horizon')
plt.tight_layout(); plt.savefig(os.path.join(PLOTS,'horizon_sweep.png'),dpi=150,bbox_inches='tight'); plt.show()
best_h=max(Hs,key=lambda h:results[h]['cov25'])
print(f'\\nVERDICT: best confident-call accuracy (@25%) at H={best_h}: {results[best_h][\"cov25\"]:.3f}')
print(f'(vs H=5: {results[5][\"cov25\"]:.3f}).  If shorter H is clearly higher, issue short-horizon alerts.')
"""))
nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.11.9"}},"nbformat":4,"nbformat_minor":5}
open("notebooks/horizon_sweep.ipynb","w").write(json.dumps(nb,indent=1))
print("Wrote notebooks/horizon_sweep.ipynb with",len(cells),"cells")
