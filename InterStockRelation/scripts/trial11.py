"""Trial 11 - Which stocks does the model look at to infer a stock? (same-bar fill-in)

For a bar t and a target stock k: hide k's bar-t features, show everyone's history and the
other 94 stocks' bar-t features, predict k's bar-t return. One shared model; the target
rotates over all 95 stocks. The target token reads the others through a single cross-
attention readout, so its attention row is a clean "who did I use" map per stock.

Saved: per-target attention (overall + per test chunk), gradient attributions (signed),
train correlations, ridge coefficients and R^2 per target.
Checks: model vs ridge R^2; attention vs correlation / ridge ranking; seed stability; drift.

  python scripts/trial11.py --seed 0          python scripts/trial11.py --smoke
"""
import os, sys, json, time, argparse, math
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(__file__))
from gpu_lease import Lease

p = argparse.ArgumentParser()
p.add_argument('--hist', type=int, default=12)        # bars of history per stock (1 hour)
p.add_argument('--dim', type=int, default=64)
p.add_argument('--heads', type=int, default=4)
p.add_argument('--ctx_layers', type=int, default=2)
p.add_argument('--epochs', type=int, default=40)
p.add_argument('--bs', type=int, default=512)
p.add_argument('--lr', type=float, default=1e-3)
p.add_argument('--patience', type=int, default=5)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--test_stride', type=int, default=4)
p.add_argument('--smoke', action='store_true')
a = p.parse_args()
if a.smoke: a.dim, a.heads, a.epochs, a.bs, a.test_stride = 16, 2, 2, 64, 400
torch.manual_seed(a.seed); np.random.seed(a.seed)
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
dev = torch.device('cuda' if torch.cuda.is_available() and not a.smoke else 'cpu')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
run = f'fillin_seed{a.seed}' + ('_smoke' if a.smoke else '')
OUT = os.path.join(ROOT, 'outputs', 'trial11'); os.makedirs(OUT, exist_ok=True)
print(f'run={run} device={dev} args={vars(a)}', flush=True)

# ---------------------------------------------------------------- data
D = np.load(os.path.join(ROOT, 'data', 'processed', 'intraday_features.npz'))
X = torch.from_numpy(D['X']); BAR = D['bar']; tickers = list(D['tickers'])
T, N, Fd = X.shape
Y = X[:, :, 0]                                            # standardized return = target
ok = np.arange(T) >= a.hist; ok &= BAR > 0                # skip day-open bar (overnight gap)
bars = np.where(ok)[0]; n = len(bars); i1, i2 = int(.70 * n), int(.85 * n)
splits = dict(train=bars[:i1], val=bars[i1:i2], test=bars[i2:][::a.test_stride])
if a.smoke: splits = dict(train=bars[:3000:7], val=bars[i1:i1 + 300], test=bars[i2:i2 + 30])
print({k: len(v) for k, v in splits.items()}, flush=True)

def make(ts, tgt):
    """ts: bar indices (B,), tgt: target stock (B,) -> x (B,N,hist,F+1), y (B,)"""
    idx = torch.from_numpy(np.stack([np.arange(t - a.hist + 1, t + 1) for t in ts]))
    x = X[idx].permute(0, 2, 1, 3)                          # (B,N,hist,F)
    flag = torch.zeros(x.shape[:3] + (1,))
    b = torch.arange(len(ts)); tg = torch.as_tensor(tgt)
    y = x[b, tg, -1, 0].clone()
    x[b, tg, -1, :] = 0.0; flag[b, tg, -1, 0] = 1.0           # hide all bar-t features of the target
    return torch.cat([x, flag], -1), y, tg

# ---------------------------------------------------------------- model
class FillIn(nn.Module):
    def __init__(s):
        super().__init__()
        d = a.dim; s.h, s.dh = a.heads, d // a.heads
        s.enc = nn.Sequential(nn.Linear(a.hist * (Fd + 1), 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        s.stock = nn.Embedding(N, d); s.is_tgt = nn.Embedding(2, d)
        layer = nn.TransformerEncoderLayer(d, a.heads, 2 * d, 0.1, 'gelu', batch_first=True, norm_first=True)
        s.ctx = nn.TransformerEncoder(layer, a.ctx_layers)
        s.q = nn.Linear(d, d); s.k = nn.Linear(d, d); s.v = nn.Linear(d, d); s.o = nn.Linear(d, d)
        s.norm = nn.LayerNorm(d)
        s.head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(s, x, tg):
        B = x.shape[0]
        z = s.enc(x.flatten(2)) + s.stock.weight[None]
        z = z + s.is_tgt((torch.arange(N, device=x.device)[None] == tg[:, None]).long())
        z = s.norm(s.ctx(z))
        zt = z[torch.arange(B), tg]                                       # target token
        q = s.q(zt).view(B, s.h, 1, s.dh); k = s.k(z).view(B, N, s.h, s.dh).transpose(1, 2)
        v = s.v(z).view(B, N, s.h, s.dh).transpose(1, 2)
        att = torch.softmax((q @ k.transpose(-1, -2)) / math.sqrt(s.dh), -1)   # (B,h,1,N)
        r = s.o((att @ v).reshape(B, -1))
        return s.head(torch.cat([zt, r], -1)).squeeze(-1), att.squeeze(2).mean(1)  # pred, (B,N)

model = FillIn().to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
print(f'params {sum(q.numel() for q in model.parameters())/1e3:.1f}k', flush=True)

# ---------------------------------------------------------------- resumable training on a GPU lease
lease = Lease(run, OUT, dev)
STATE = os.path.join(OUT, f'{run}_state.pt'); CK = os.path.join(OUT, f'{run}.pt')
S = dict(ep=0, step=0, tot=0.0, nb=0, best=1e9, bad=0, hist=[], phase='train')
if os.path.exists(STATE):
    st = torch.load(STATE, map_location=dev, weights_only=False)
    model.load_state_dict(st['model']); opt.load_state_dict(st['opt']); S = st['S']
    print(f"resumed at epoch {S['ep']} step {S['step']}", flush=True)
def save_state():
    torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), S=S), STATE + '.tmp'); os.replace(STATE + '.tmp', STATE)

def val_loss(ts):
    model.eval(); rng = np.random.default_rng(123); tot = 0.0
    with torch.no_grad():
        for i in range(0, len(ts), a.bs):
            t = ts[i:i + a.bs]; x, y, tg = make(t, rng.integers(0, N, len(t)))
            pr, _ = model(x.to(dev), tg.to(dev)); tot += ((pr - y.to(dev)) ** 2).sum().item()
    return tot / len(ts)

while S['phase'] == 'train' and S['ep'] < a.epochs:
    rng = np.random.default_rng(a.seed * 1000 + S['ep'])
    order = rng.permutation(splits['train']); tgts = rng.integers(0, N, len(order))
    nsteps = (len(order) + a.bs - 1) // a.bs; t0 = time.time(); model.train()
    for s in range(S['step'], nsteps):
        sl = slice(s * a.bs, (s + 1) * a.bs); x, y, tg = make(order[sl], tgts[sl])
        pr, _ = model(x.to(dev), tg.to(dev)); L = ((pr - y.to(dev)) ** 2).mean()
        opt.zero_grad(set_to_none=True); L.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        S['tot'] += L.item(); S['nb'] += 1; S['step'] = s + 1
        if S['step'] % 50 == 0:
            lease.beat(); why = lease.should_yield()
            if why: lease.release(save_state, why)
    vl = val_loss(splits['val']); tl = S['tot'] / max(S['nb'], 1); S['hist'].append((tl, vl))
    print(f"ep {S['ep']:02d} train {tl:.4f} val {vl:.4f} ({time.time()-t0:.0f}s, tier {lease.tier})", flush=True)
    if vl < S['best']: S['best'], S['bad'] = vl, 0; torch.save(model.state_dict(), CK)
    else: S['bad'] += 1
    S.update(ep=S['ep'] + 1, step=0, tot=0.0, nb=0)
    if S['bad'] >= a.patience: print('early stop', flush=True); S['ep'] = a.epochs
    save_state(); lease.beat(); why = lease.should_yield()
    if why: lease.release(save_state, why)
S['phase'] = 'eval'; save_state()
model.load_state_dict(torch.load(CK, map_location=dev)); model.eval()

# ---------------------------------------------------------------- evaluation: every test bar x every target
te = splits['test']; nch = 3; chunk_of = np.minimum(np.arange(len(te)) * nch // len(te), nch - 1)
A = np.zeros((nch, N, N)); cnt = np.zeros((nch, N)); sse = np.zeros(N); sst = np.zeros(N)
G = np.zeros((N, N)); Gabs = np.zeros((N, N)); gcount = 0
BB = max(1, 1536 // N)                                     # bars per batch (x95 targets)
for i in range(0, len(te), BB):
    t = te[i:i + BB]; tt = np.repeat(t, N); tg = np.tile(np.arange(N), len(t))
    x, y, tgv = make(tt, tg); x, y, tgv = x.to(dev), y.to(dev), tgv.to(dev)
    want_grad = gcount < 400
    if want_grad: x.requires_grad_(True)
    with torch.set_grad_enabled(want_grad):
        pr, att = model(x, tgv)
        if want_grad:
            pr.sum().backward(); g = x.grad[:, :, -1, 0].detach().view(len(t), N, N)   # d pred_k / d ret_j(t)
            G += g.sum(0).cpu().numpy(); Gabs += g.abs().sum(0).cpu().numpy(); gcount += len(t)
    att = att.detach().view(len(t), N, N).cpu().numpy(); err = ((pr - y) ** 2).detach().view(len(t), N).cpu().numpy()
    yy = y.view(len(t), N).cpu().numpy()
    for c in range(nch):
        m = chunk_of[i:i + BB] == c
        if m.any(): A[c] += att[m].sum(0); cnt[c] += m.sum()
    sse += err.sum(0); sst += (yy ** 2).sum(0)
A_chunk = A / cnt[:, :, None]; A_all = A.sum(0) / cnt.sum(0)[:, None]
r2_model = 1 - sse / sst

# baselines and references (train period only)
Rtr = D['ret'][splits['train']]; Ctr = np.corrcoef(Rtr, rowvar=False)
Ytr = Y[splits['train']].numpy(); Yte = Y[te].numpy()
coef = np.zeros((N, N)); r2_ridge = np.zeros(N)
for k in range(N):
    o = np.r_[0:k, k + 1:N]; Xa = Ytr[:, o]; lam = 1.0 * len(Xa)
    w = np.linalg.solve(Xa.T @ Xa + lam * 1e-3 * np.eye(N - 1), Xa.T @ Ytr[:, k])
    coef[k, o] = w; pred = Yte[:, o] @ w
    r2_ridge[k] = 1 - ((Yte[:, k] - pred) ** 2).sum() / (Yte[:, k] ** 2).sum()

def rank_corr(u, v):
    ru, rv = np.argsort(np.argsort(u)), np.argsort(np.argsort(v)); return np.corrcoef(ru, rv)[0, 1]
off = lambda M, k: np.r_[M[k, :k], M[k, k + 1:]]
sp_corr = np.array([rank_corr(off(A_all, k), off(Ctr, k)) for k in range(N)])
sp_ridge = np.array([rank_corr(off(A_all, k), off(np.abs(coef), k)) for k in range(N)])
top5 = lambda M, k: set(np.argsort(-np.r_[M[k, :k], [-np.inf], M[k, k + 1:]])[:5])
ov5 = np.array([len(top5(A_all, k) & top5(Ctr, k)) / 5 for k in range(N)])
Gm = G / max(gcount, 1)
sign_agree = float(np.mean([(np.sign(off(Gm, k)) == np.sign(off(coef, k))).mean() for k in range(N)]))
# drift: does attention change across test chunks in the same direction as realized correlation?
Rte = D['ret'][te]; Cch = [np.corrcoef(Rte[chunk_of == c], rowvar=False) for c in range(nch)]
iu = ~np.eye(N, dtype=bool)
dA = (A_chunk[-1] - A_chunk[0])[iu]; dC = (Cch[-1] - Cch[0])[iu]
drift_corr = float(np.corrcoef(dA, dC)[0, 1])

res = dict(run=run, args=vars(a), epochs_run=len(S['hist']), best_val=S['best'],
    r2=dict(model_mean=float(r2_model.mean()), ridge_mean=float(r2_ridge.mean()),
            model_beats_ridge_frac=float((r2_model > r2_ridge).mean())),
    attention_vs_corr=dict(spearman_mean=float(sp_corr.mean()), top5_overlap_mean=float(ov5.mean()),
                           spearman_vs_ridge_abs=float(sp_ridge.mean()), self_weight_mean=float(np.diag(A_all).mean())),
    gradient_sign_agree_with_ridge=sign_agree, drift_attention_vs_corr_change=drift_corr,
    examples={tickers[k]: [tickers[j] for j in np.argsort(-np.r_[A_all[k, :k], [-1], A_all[k, k + 1:]])[:5]]
              for k in [tickers.index(s) for s in ['CANBK', 'TCS', 'TATASTEEL', 'HDFCBANK', 'RELIANCE', 'SUNPHARMA'] if s in tickers]})
json.dump(res, open(os.path.join(OUT, f'{run}.json'), 'w'), indent=1)
np.savez_compressed(os.path.join(OUT, f'{run}_maps.npz'), A_all=A_all, A_chunk=A_chunk, G=Gm, Gabs=Gabs / max(gcount, 1),
                    C_train=Ctr, C_chunks=np.stack(Cch), ridge_coef=coef, r2_model=r2_model, r2_ridge=r2_ridge, tickers=np.array(tickers))
print(json.dumps({k: v for k, v in res.items() if k != 'args'}, indent=1), flush=True)
lease.finish('isr_t11_')
