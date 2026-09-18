import numpy as np, pickle, os
DATA='data/processed'
aux=np.load(os.path.join(DATA,'aux_corr_videos.npy'), mmap_mode='r')  # (1157,11,95,95)
meta=pickle.load(open(os.path.join(DATA,'aux_corr_meta.pkl'),'rb'))
cols=meta['channel_names']
H=5; LOOKBACK=20; N=95
iu,ju=np.triu_indices(N,k=1)

def mag_stratified(video):
    T=video.shape[0]
    cs=np.cumsum(video,0)
    true=[]; pred=[]; mag=[]
    for t in range(LOOKBACK, T-H):
        tm=(cs[t-1]-cs[t-1-LOOKBACK])/LOOKBACK
        ch=(video[t+H]-video[t])[iu,ju]
        pr=-np.sign((video[t]-tm))[iu,ju]         # level mean-reversion
        true.append(np.sign(ch)); pred.append(pr); mag.append(np.abs(ch))
    t=np.concatenate(true); p=np.concatenate(pred); m=np.concatenate(mag)
    out={}
    for tau in [0.0,0.15,0.30]:
        sel=(m>tau)&(t!=0)
        out[tau]=float((p[sel]==t[sel]).mean())
    return out

print("EWMA CONFOUND CHECK: does magnitude-predictability hold on RAW windowed corr?")
print(f"{'correlation type':<22}{'dir@0':>8}{'dir@0.15':>10}{'dir@0.30':>10}")
print('-'*50)
for name in ['R_short','R_single','R_medium','R_long','R_ewma']:
    ci=cols.index(name)
    v=np.nan_to_num(np.asarray(aux[:,ci]), nan=0.0)
    r=mag_stratified(v)
    tag=' (EWMA=corr_video)' if name=='R_ewma' else ' (raw windowed)'
    print(f"{name+tag:<22}{r[0.0]:>8.3f}{r[0.15]:>10.3f}{r[0.30]:>10.3f}")
print("\nIf the rise with magnitude holds on raw windowed corr too -> NOT an EWMA artifact.")
