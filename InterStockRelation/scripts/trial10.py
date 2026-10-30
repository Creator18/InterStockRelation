"""Trial 10 - Looped transformer with learned, input-dependent time windows and
learned halting, over raw 5-min stock tokens (no images).

Per loop pass l (one shared block):
  1. temporal self-attention within each stock; each stock/head picks its own
     look-back window w from its current latent (soft ramp mask, causal)
  2. cross-stock attention over the 95 per-stock summaries (learned spatial attention)
  3. FFN, then a halting head (PonderNet) decides whether to stop
Targets: next-H-bar return of every stock, and the realized correlation matrix
over the masked next-H window. Window cost + halting prior control the lookup.

Modes:  --mode learned | fixed (schedule 3,6,12,24,ctx) | full (no window)
        --halting 1|0
Usage:  python scripts/trial10.py --mode learned --halting 1
        python scripts/trial10.py --smoke        (CPU sanity run)
"""
import os, json, time, argparse, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument('--mode', default='learned', choices=['learned', 'fixed', 'full', 'fixedw'])
p.add_argument('--window', type=int, default=12)       # fixedw: same window on every pass
p.add_argument('--target', default='level', choices=['level', 'delta'])  # delta: predict change over current corr
p.add_argument('--dir_w', type=float, default=0.5)     # direction loss weight (delta target)
p.add_argument('--halting', type=int, default=1)
p.add_argument('--ctx', type=int, default=150)        # 2 trading days of context
p.add_argument('--H', type=int, default=24)           # 2-hour masked future window
p.add_argument('--stride', type=int, default=3)
p.add_argument('--dim', type=int, default=64)
p.add_argument('--heads', type=int, default=4)
p.add_argument('--loops', type=int, default=5)
p.add_argument('--epochs', type=int, default=30)
p.add_argument('--bs', type=int, default=16)
p.add_argument('--lr', type=float, default=3e-4)
p.add_argument('--span_cost', type=float, default=0.02)
p.add_argument('--kl_w', type=float, default=0.01)
p.add_argument('--prior', type=float, default=0.3)    # geometric halting prior
p.add_argument('--patience', type=int, default=6)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--smoke', action='store_true')
p.add_argument('--tag', default='')
a = p.parse_args()
if a.smoke:
    a.ctx, a.H, a.dim, a.heads, a.loops, a.bs, a.stride = 30, 12, 16, 2, 3, 4, 60
    if a.epochs == 30: a.epochs = 1
torch.manual_seed(a.seed); np.random.seed(a.seed)
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True   # big speedup on A100/H100
dev = torch.device('cuda' if torch.cuda.is_available() and not a.smoke else 'cpu')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
run = a.tag or f"{a.mode}_halt{a.halting}" + ('_smoke' if a.smoke else '')
OUT = os.environ.get('ISR_OUT', os.path.join(ROOT, 'outputs', 'trial10')); os.makedirs(OUT, exist_ok=True)
print(f'run={run} device={dev} args={vars(a)}', flush=True)

# ---------------------------------------------------------------- data
D = np.load(os.path.join(ROOT, 'data', 'processed', 'intraday_features.npz'))
X = torch.from_numpy(D['X']); RET = torch.from_numpy(D['ret'])
BAR = torch.from_numpy(D['bar'].astype(np.int64)); DAY = D['day']
T, N, Fdim = X.shape
t_all = np.arange(a.ctx - 1, T - a.H)
same_day = DAY[t_all + a.H] == DAY[t_all]          # future window must not cross the close
t_all = t_all[same_day][::a.stride]
n = len(t_all); i1, i2 = int(.70 * n), int(.85 * n)
splits = {'train': t_all[:i1], 'val': t_all[i1:i2], 'test': t_all[i2:]}
if a.smoke: splits = {k: v[:40] for k, v in splits.items()}
print({k: len(v) for k, v in splits.items()}, flush=True)

def corr_rows(r):                                   # r: (B, L, N) -> (B, N, N)
    r = r - r.mean(1, keepdim=True)
    c = r.transpose(1, 2) @ r
    d = torch.sqrt(torch.diagonal(c, dim1=1, dim2=2).clamp_min(1e-12))
    return (c / (d[:, :, None] * d[:, None, :])).clamp(-1, 1)

tr_end = splits['train'][-1]
LONGRUN = corr_rows(RET[None, :tr_end]).squeeze(0)  # train-period correlation

def batch(ts):
    idx = torch.from_numpy(np.stack([np.arange(t - a.ctx + 1, t + 1) for t in ts]))
    fidx = torch.from_numpy(np.stack([np.arange(t + 1, t + a.H + 1) for t in ts]))
    x = X[idx]                                      # (B, C, N, F)
    bar = BAR[idx]                                  # (B, C)
    fut = RET[fidx]                                 # (B, H, N)
    past = RET[idx][:, -a.H:]                       # last H bars of context
    return dict(x=x, bar=bar, fret=fut.sum(1) / (RET[:tr_end].std(0) * math.sqrt(a.H)),
                fcorr=corr_rows(fut), ccorr=corr_rows(past), hcorr=corr_rows(RET[idx]),
                past_absret=RET[idx].abs().mean((1, 2)))

# ---------------------------------------------------------------- model
class LoopedStockTransformer(nn.Module):
    def __init__(s):
        super().__init__()
        d, h = a.dim, a.heads; s.h, s.dh = h, d // h
        s.inp = nn.Linear(Fdim, d); s.stock = nn.Embedding(N, d); s.tod = nn.Embedding(80, d)
        s.loop_emb = nn.Embedding(a.loops, d)
        s.n1, s.n2, s.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        s.qkv_t = nn.Linear(d, 3 * d); s.o_t = nn.Linear(d, d)
        s.qkv_x = nn.Linear(d, 3 * d); s.o_x = nn.Linear(d, d)
        s.ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        s.span = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, h))   # window chooser
        s.halt = nn.Linear(d, 1)
        s.ret_head = nn.Linear(d, 1); s.corr_head = nn.Linear(d, d)
        s.wc = nn.Parameter(torch.zeros(())); s.wl = nn.Parameter(torch.zeros(())); s.bc = nn.Parameter(torch.zeros(())); s.g = nn.Parameter(torch.zeros(()))   # start at 'no change'
        s.ramp = 4.0
        sched = [3, 6, 12, 24, a.ctx]
        s.fixed = [sched[min(l, len(sched) - 1)] for l in range(a.loops)]

    def window(s, hl, l):                            # hl: (B, N, d) -> w: (B, N, h)
        if a.mode == 'fixed': return torch.full(hl.shape[:2] + (s.h,), float(s.fixed[l]), device=hl.device)
        if a.mode == 'fixedw': return torch.full(hl.shape[:2] + (s.h,), float(a.window), device=hl.device)
        if a.mode == 'full':  return torch.full(hl.shape[:2] + (s.h,), float(a.ctx), device=hl.device)
        return 2.0 + (a.ctx - 2.0) * torch.sigmoid(s.span(hl))

    def temporal(s, z, w):                          # z: (B, N, C, d)
        B, Nn, C, d = z.shape
        q, k, v = s.qkv_t(z).view(B, Nn, C, 3, s.h, s.dh).permute(3, 0, 1, 4, 2, 5)  # (B,N,h,C,dh)
        sc = (q @ k.transpose(-1, -2)) / math.sqrt(s.dh)                               # (B,N,h,C,C)
        pos = torch.arange(C, device=z.device)
        lag = (pos[:, None] - pos[None, :]).float()                                    # q - k
        sc = sc.masked_fill(lag < 0, float('-inf'))
        att = torch.softmax(sc, -1)
        m = ((w[..., None, None] - lag) / s.ramp + 1).clamp(0, 1)                      # soft window
        att = att * m; att = att / att.sum(-1, keepdim=True).clamp_min(1e-9)
        out = (att @ v).permute(0, 1, 3, 2, 4).reshape(B, Nn, C, d)
        return s.o_t(out)

    def cross(s, hl):                               # hl: (B, N, d)
        B, Nn, d = hl.shape
        q, k, v = s.qkv_x(hl).view(B, Nn, 3, s.h, s.dh).permute(2, 0, 3, 1, 4)        # (B,h,N,dh)
        alpha = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(s.dh), -1)          # (B,h,N,N)
        out = (alpha @ v).transpose(1, 2).reshape(B, Nn, d)
        return s.o_x(out), alpha, v

    def heads(s, hl, cc):
        r = s.ret_head(hl).squeeze(-1)
        if a.target == 'delta':                     # change over the current (last-H-bar) correlation
            u = s.corr_head(hl) / math.sqrt(a.dim)
            lr = LR_DEV[None]
            dlt = torch.tanh(s.g * (u @ u.transpose(1, 2)) + s.wc * cc + s.wl * (lr - cc) + s.bc)
            return r, (cc + dlt).clamp(-1, 1)
        e = F.normalize(s.corr_head(hl), dim=-1)
        return r, e @ e.transpose(1, 2)

    def forward(s, x, bar, cc, keep=False):
        B, C, Nn, _ = x.shape
        z = s.inp(x).permute(0, 2, 1, 3)                                   # (B,N,C,d)
        z = z + s.stock.weight[None, :, None] + s.tod(bar)[:, None]
        outs, lams, ws, rec = [], [], [], []
        for l in range(a.loops):
            z = z + s.loop_emb.weight[l]
            hl = z[:, :, -1]
            w = s.window(s.n1(hl), l)
            z = z + s.temporal(s.n1(z), w)
            xo, alpha, v = s.cross(s.n2(z[:, :, -1]))
            z = z + xo[:, :, None]                                           # broadcast to sequence
            z = z + s.ffn(s.n3(z))
            hl = z[:, :, -1]
            outs.append(s.heads(hl, cc)); ws.append(w.mean(-1))
            lams.append(torch.sigmoid(s.halt(hl)).squeeze(-1))
            if keep: rec.append((alpha, v))
        # PonderNet halting distribution per stock
        L = a.loops
        if a.halting:
            lam = torch.stack(lams, 1); lam[:, -1] = 1.0                    # (B,L,N)
            notyet = torch.cumprod(torch.cat([torch.ones_like(lam[:, :1]), 1 - lam[:, :-1]], 1), 1)
            ph = lam * notyet
        else:
            ph = torch.zeros(B, L, Nn, device=x.device); ph[:, -1] = 1.0
        return outs, ph, torch.stack(ws, 1), rec

LR_DEV = LONGRUN.to(dev)
model = LoopedStockTransformer().to(dev)
print(f'params: {sum(q.numel() for q in model.parameters())/1e3:.1f}k', flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
prior = torch.tensor([a.prior * (1 - a.prior) ** l for l in range(a.loops)]); prior = (prior / prior.sum()).to(dev)
offd = ~torch.eye(N, dtype=torch.bool, device=dev)

def expected(outs, ph):
    r = sum(ph[:, l] * outs[l][0] for l in range(len(outs)))
    c = sum(ph[:, l, :, None] * outs[l][1] for l in range(len(outs)))
    return r, 0.5 * (c + c.transpose(1, 2))

def loss_fn(outs, ph, ws, b):
    tot = 0.0
    for l, (r, c) in enumerate(outs):
        lr_ = (r - b['fret']) ** 2                                           # (B,N)
        lc = (((c - b['fcorr']) ** 2) * offd).sum(-1) / (N - 1)              # (B,N) per stock row
        if a.target == 'delta':                     # smooth sign-agreement on meaningful changes
            dt = b['fcorr'] - b['ccorr']; dp = c - b['ccorr']; mk = (dt.abs() > 0.05) & offd
            ld = (((1 - torch.tanh(20 * dp) * torch.sign(dt)) / 2) * mk).sum(-1) / mk.sum(-1).clamp_min(1)
            lc = lc + a.dir_w * ld
        tot = tot + (ph[:, l] * (lr_ + 4.0 * lc)).mean()
    span = (ph * ws / a.ctx).sum(1).mean() if a.mode == 'learned' else torch.tensor(0., device=dev)
    kl = (ph.clamp_min(1e-9) * (ph.clamp_min(1e-9).log() - prior.log()[None, :, None])).sum(1).mean() if a.halting else torch.tensor(0., device=dev)
    return tot + a.span_cost * span + a.kl_w * kl

def todev(b): return {k: v.to(dev) for k, v in b.items()}

# ---------------------------------------------------------------- GPU lease: start anywhere, hand off to a better GPU
# One "owner" job trains at a time. A job that starts on a better GPU asks the owner to
# checkpoint and release; it then resumes from the exact step. Effective batch size is
# fixed (micro-batch x accumulation), so results do not depend on which GPU ran them.
JOB = os.environ.get('SLURM_JOB_ID', f'local{os.getpid()}')
LEASE = os.path.join(OUT, f'{run}.lease.json'); STATE = os.path.join(OUT, f'{run}_state.pt'); ck = os.path.join(OUT, f'{run}.pt')
def gpu_tier():
    if dev.type != 'cuda': return 0
    pr = torch.cuda.get_device_properties(0); nm = pr.name.upper()
    rank = next((r for k, r in [('H200', 9), ('H100', 8), ('A100', 7), ('L40', 6), ('A6000', 5), ('A5000', 4),
                                ('V100', 3), ('T4', 2), ('P100', 1)] if k in nm), 1)
    return rank * 1000 + pr.total_memory // 2**30
def rd():
    try: return json.load(open(LEASE))
    except Exception: return {}
def wr(d):
    tmp = LEASE + f'.{JOB}'; json.dump(d, open(tmp, 'w')); os.replace(tmp, LEASE)
TIER = gpu_tier(); STALE = 900
L0 = rd()
if L0.get('state') == 'done': print('run already finished', flush=True); raise SystemExit(0)
if L0.get('state') == 'running' and time.time() - L0.get('beat', 0) < STALE and L0.get('job') != JOB:
    if TIER <= L0.get('tier', 0): print(f"owner {L0['job']} has an equal/better GPU; exiting", flush=True); raise SystemExit(0)
    L0['request'] = dict(job=JOB, tier=TIER); wr(L0); print(f"requesting takeover from {L0['job']}", flush=True)
    t0 = time.time()
    while time.time() - t0 < 1800:
        cur = rd()
        if cur.get('state') != 'running' or time.time() - cur.get('beat', 0) > STALE: break
        time.sleep(15)
def beat(state='running'):
    req = rd().get('request')                       # keep a pending upgrade request intact
    d = dict(job=JOB, tier=TIER, state=state, beat=time.time(), gpu=torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu')
    if req and req.get('job') != JOB and state == 'running': d['request'] = req
    wr(d)
beat(); print(f'lease acquired job={JOB} tier={TIER}', flush=True)

# micro-batch sized to this GPU, accumulation keeps effective batch = a.bs
MICRO = a.bs
if dev.type == 'cuda' and not a.smoke:
    torch.cuda.reset_peak_memory_stats(); bb = todev(batch(splits['train'][:1]))
    o_, ph_, ws_, _ = model(bb['x'], bb['bar'], bb['ccorr']); loss_fn(o_, ph_, ws_, bb).backward(); model.zero_grad(set_to_none=True)
    per = torch.cuda.max_memory_allocated(); free = torch.cuda.get_device_properties(0).total_memory * 0.85
    MICRO = max(1, min(a.bs, int(free // per)))
    while a.bs % MICRO: MICRO -= 1
    del bb, o_, ph_, ws_; torch.cuda.empty_cache()
ACCUM = a.bs // MICRO
print(f'micro-batch {MICRO} x accum {ACCUM} = effective {a.bs}', flush=True)

stop = {'flag': False}
import signal
if hasattr(signal, 'SIGUSR1'): signal.signal(signal.SIGUSR1, lambda *_: stop.update(flag=True))   # Slurm time-limit warning
signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))

S = dict(ep=0, step=0, tot=0.0, nb=0, best=1e9, bad=0, hist=[], phase='train')
if os.path.exists(STATE):
    st = torch.load(STATE, map_location=dev, weights_only=False)
    model.load_state_dict(st['model']); opt.load_state_dict(st['opt']); S = st['S']
    print(f"resumed at epoch {S['ep']} step {S['step']} (phase {S['phase']})", flush=True)

def save_state():
    torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), S=S), STATE + '.tmp'); os.replace(STATE + '.tmp', STATE)

def must_yield():
    if stop['flag']: return 'signal'
    r = rd().get('request')
    return 'upgrade' if r and r.get('tier', 0) > TIER else None

def release(reason):
    save_state(); cur = rd(); cur.update(state='released', beat=time.time()); cur.pop('request', None); wr(cur)
    print(f'checkpointed and released lease ({reason})', flush=True)
    if reason == 'signal' and 'SLURM_JOB_ID' in os.environ: os.system(f'scontrol requeue {JOB}')
    raise SystemExit(0)

def evaluate_loss(ts):
    model.eval(); tot, nb = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(ts), MICRO):
            b = todev(batch(ts[i:i + MICRO])); o, ph, ws, _ = model(b['x'], b['bar'], b['ccorr'])
            tot += loss_fn(o, ph, ws, b).item() * len(b['x']); nb += len(b['x'])
    return tot / max(nb, 1)

while S['phase'] == 'train' and S['ep'] < a.epochs:
    order = np.random.default_rng(a.seed + S['ep']).permutation(splits['train'])   # same order on resume
    steps = range(S['step'], (len(order) + a.bs - 1) // a.bs); t0 = time.time(); model.train()
    for s in steps:
        opt.zero_grad(set_to_none=True); chunk = order[s * a.bs:(s + 1) * a.bs]
        for m0 in range(0, len(chunk), MICRO):
            b = todev(batch(chunk[m0:m0 + MICRO])); o, ph, ws, _ = model(b['x'], b['bar'], b['ccorr'])
            L = loss_fn(o, ph, ws, b) * len(b['x']) / len(chunk); L.backward(); S['tot'] += L.item()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); S['nb'] += 1; S['step'] = s + 1
        if S['step'] % 50 == 0:
            beat(); why = must_yield()
            if why: release(why)
    vl = evaluate_loss(splits['val']); tl = S['tot'] / max(S['nb'], 1)
    S['hist'].append((tl, vl)); print(f"ep {S['ep']:02d} train {tl:.4f} val {vl:.4f} ({time.time()-t0:.0f}s, {TIER})", flush=True)
    if vl < S['best']: S['best'], S['bad'] = vl, 0; torch.save(model.state_dict(), ck)
    else: S['bad'] += 1
    S.update(ep=S['ep'] + 1, step=0, tot=0.0, nb=0)
    if S['bad'] >= a.patience: print('early stop', flush=True); S['ep'] = a.epochs
    save_state(); beat()
    why = must_yield()
    if why: release(why)
S['phase'] = 'eval'; save_state()
best, hist = S['best'], S['hist']
model.load_state_dict(torch.load(ck, map_location=dev))

# ---------------------------------------------------------------- evaluation
model.eval()
IU = torch.triu_indices(N, N, 1, device=dev); predsave, cursave, trusave = [], [], []
acc = {k: [] for k in ['pm', 'cur', 'tru', 'hist', 'lr', 'rp', 'rt', 'w', 'depth', 'vol', 'sgn_attr', 'sgn_true']}
LR = LONGRUN.to(dev)
with torch.no_grad():
    for i in range(0, len(splits['test']), MICRO):
        b = todev(batch(splits['test'][i:i + MICRO]))
        outs, ph, ws, rec = model(b['x'], b['bar'], b['ccorr'], keep=True)
        r, c = expected(outs, ph)
        m = offd[None].expand_as(c)
        predsave.append(c[:, IU[0], IU[1]].half().cpu()); cursave.append(b['ccorr'][:, IU[0], IU[1]].half().cpu()); trusave.append(b['fcorr'][:, IU[0], IU[1]].half().cpu())
        acc['pm'].append(c[m].cpu()); acc['cur'].append(b['ccorr'][m].cpu()); acc['tru'].append(b['fcorr'][m].cpu())
        acc['hist'].append(b['hcorr'][m].cpu()); acc['lr'].append(LR[None].expand_as(c)[m].cpu())
        acc['rp'].append(r.cpu()); acc['rt'].append(b['fret'].cpu())
        acc['w'].append((ph * ws).sum(1).cpu())
        acc['depth'].append((ph * torch.arange(1, a.loops + 1, device=dev)[None, :, None]).sum(1).cpu())
        acc['vol'].append(b['past_absret'].cpu())
        # signed contribution of stock j to stock i's return prediction (last loop, linearized)
        alpha, v = rec[-1]                                                  # (B,h,N,N), (B,h,N,dh)
        Wo = model.o_x.weight.view(a.dim, a.heads, a.dim // a.heads)        # (d, h, dh)
        proj = torch.einsum('d,dhk,bhjk->bhj', model.ret_head.weight[0], Wo, v)   # value pushed toward return
        contrib = (alpha * proj[:, :, None, :]).sum(1)                       # (B,N,N) c_ij
        move = torch.sign(b['x'][:, -6:, :, 0].sum(1))                       # j's recent direction
        rel = torch.sign(contrib) * move[:, None, :]
        acc['sgn_attr'].append(rel[m].cpu()); acc['sgn_true'].append(torch.sign(b['fcorr'])[m].cpu())

cat = {k: torch.cat(v).numpy() for k, v in acc.items()}
def dir_acc(pred, cur, tru, sel=None):
    dt = tru - cur; dp = pred - cur; ok = np.abs(dt) > 1e-6
    if sel is not None: ok &= sel
    return float((np.sign(dp[ok]) == np.sign(dt[ok])).mean())
pm, cur, tru = cat['pm'], cat['cur'], cat['tru']
mag = np.abs(pm - cur); q75, q90 = np.quantile(mag, .75), np.quantile(mag, .90)
res = dict(run=run, args=vars(a), epochs_run=len(hist), best_val=best,
    corr_mse=dict(model=float(((pm - tru) ** 2).mean()), persistence=float(((cur - tru) ** 2).mean()),
                  historical_ctx=float(((cat['hist'] - tru) ** 2).mean()), longrun=float(((cat['lr'] - tru) ** 2).mean())),
    corr_dir=dict(model=dir_acc(pm, cur, tru), mean_reversion_rule=dir_acc(cat['lr'], cur, tru),
                  historical_ctx=dir_acc(cat['hist'], cur, tru),
                  model_top25=dir_acc(pm, cur, tru, mag >= q75), model_top10=dir_acc(pm, cur, tru, mag >= q90),
                  # fair selective baseline: mean-reversion rule on its own most confident 25% / 10%
                  mean_reversion_top25=dir_acc(cat['lr'], cur, tru, np.abs(cat['lr'] - cur) >= np.quantile(np.abs(cat['lr'] - cur), .75)),
                  mean_reversion_top10=dir_acc(cat['lr'], cur, tru, np.abs(cat['lr'] - cur) >= np.quantile(np.abs(cat['lr'] - cur), .90))),
    ret=dict(mse_model=float(((cat['rp'] - cat['rt']) ** 2).mean()), mse_zero=float((cat['rt'] ** 2).mean()),
             ic=float(np.nanmean([np.corrcoef(x, y)[0, 1] for x, y in zip(cat['rp'], cat['rt'])])),
             dir=float((np.sign(cat['rp']) == np.sign(cat['rt'])).mean())),
    windows=dict(mean_bars=float(cat['w'].mean()), p10=float(np.quantile(cat['w'], .1)), p90=float(np.quantile(cat['w'], .9)),
                 corr_with_vol=float(np.corrcoef(cat['w'].mean(1), cat['vol'])[0, 1])),
    halting=dict(mean_depth=float(cat['depth'].mean()), corr_with_vol=float(np.corrcoef(cat['depth'].mean(1), cat['vol'])[0, 1])),
    attribution=dict(sign_agree_all=float((cat['sgn_attr'] == cat['sgn_true']).mean())))
strong = np.abs(tru) > 0.3
res['attribution']['sign_agree_strong_pairs'] = float((cat['sgn_attr'][strong] == cat['sgn_true'][strong]).mean())
json.dump(res, open(os.path.join(OUT, f'{run}.json'), 'w'), indent=1)
# per-sample test predictions (upper triangle) for the oracle window-chooser analysis
np.save(os.path.join(OUT, f'{run}_pred.npy'), torch.cat(predsave).numpy())
if not os.path.exists(os.path.join(OUT, 'test_cur.npy')):
    np.save(os.path.join(OUT, 'test_cur.npy'), torch.cat(cursave).numpy()); np.save(os.path.join(OUT, 'test_tru.npy'), torch.cat(trusave).numpy())
print(json.dumps({k: v for k, v in res.items() if k != 'args'}, indent=1))
beat('done')
# training finished: cancel any still-pending upgrade jobs for this run
if 'SLURM_JOB_ID' in os.environ: os.system(f"scancel --me --name=isr_t10_{run} --state=PENDING")
