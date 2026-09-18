import numpy as np, pickle, os
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
DATA='data/processed'
corr=np.load(os.path.join(DATA,'corr_video.npy'))
vm=pickle.load(open(os.path.join(DATA,'video_meta.pkl'),'rb'))
ordered=vm['ordered_tickers']
T,N,_=corr.shape; K=8

def cluster(frames):
    mc=frames.mean(0)
    return KMeans(n_clusters=K,n_init=10,random_state=42).fit_predict(mc)

# split timeline into thirds
t1=corr[:T//3]; t2=corr[T//3:2*T//3]; t3=corr[2*T//3:]
L1=cluster(t1); L2=cluster(t2); L3=cluster(t3); Lall=cluster(corr)

print("Cluster stability across time (Adjusted Rand Index; 1.0=identical, 0=random):")
print(f"  1st third vs 2nd third : {adjusted_rand_score(L1,L2):.3f}")
print(f"  2nd third vs 3rd third : {adjusted_rand_score(L2,L3):.3f}")
print(f"  1st third vs 3rd third : {adjusted_rand_score(L1,L3):.3f}")
print(f"  full-sample vs recent  : {adjusted_rand_score(Lall,L3):.3f}")

# how many stocks keep the same cluster-mates (co-membership stability)
def comembership(L):
    return (L[:,None]==L[None,:])
cm1,cm3=comembership(L1),comembership(L3)
iu,ju=np.triu_indices(N,k=1)
agree=(cm1[iu,ju]==cm3[iu,ju]).mean()
print(f"\n  Pairwise co-membership agreement (1st vs 3rd third): {agree:.3f}")
print("  (fraction of stock pairs that are together-or-apart consistently across time)")
print("\nStable if ARI stays well above 0 and co-membership agreement is high (>0.85).")
