"""Window-oracle analysis for the Trial 10 delta sweep.

Given test predictions from one model per fixed window, how much would a PERFECT window
chooser gain over the best single window? That is the ceiling for any learned chooser.
  per-sample oracle : pick the best window for each test sample (one window for all pairs)
  per-pair oracle   : pick the best window for each (sample, pair) - a loose upper bound
Compared against the learned-window model and the mean-reversion rule.

  python scripts/analyze_window_oracle.py <dir with *_pred.npy, test_cur.npy, test_tru.npy, *.json>
"""
import sys, os, json, glob
import numpy as np

d = sys.argv[1] if len(sys.argv) > 1 else os.path.expandvars('/scratch/$USER/isr-out/trial10_windows')
cur = np.load(os.path.join(d, 'test_cur.npy')).astype(np.float32)
tru = np.load(os.path.join(d, 'test_tru.npy')).astype(np.float32)
dt = tru - cur; ok = np.abs(dt) > 1e-6

def dir_acc(pred, mask=None):
    m = ok if mask is None else ok & mask
    return float((np.sign(pred - cur)[m] == np.sign(dt)[m]).mean())

wins = [3, 6, 12, 24, 75, 150]
P = {w: np.load(os.path.join(d, f'fixedw{w}_delta_pred.npy')).astype(np.float32) for w in wins
     if os.path.exists(os.path.join(d, f'fixedw{w}_delta_pred.npy'))}
learned = os.path.join(d, 'learned_halt1_delta_nospan_pred.npy')
rows = []
for w, p in P.items():
    rows.append((f'fixed window {w:>3} bars', float(((p - tru) ** 2).mean()), dir_acc(p)))
if os.path.exists(learned):
    p = np.load(learned).astype(np.float32); rows.append(('learned windows (no span cost)', float(((p - tru) ** 2).mean()), dir_acc(p)))
rows.append(('persistence (no change)', float((dt ** 2).mean()), float('nan')))

ws = list(P); stack = np.stack([P[w] for w in ws])                  # (W, S, pairs)
err = (stack - tru[None]) ** 2
best_single = min(ws, key=lambda w: ((P[w] - tru) ** 2).mean())
# per-sample oracle: one window per test sample, chosen in hindsight by that sample's MSE
pick_s = err.mean(2).argmin(0)                                       # (S,)
orc_s = stack[pick_s, np.arange(stack.shape[1])]
# per-pair oracle: best window for every (sample, pair) - by direction correctness
right = (np.sign(stack - cur[None]) == np.sign(dt)[None])
orc_p_dir = float((right.any(0))[ok].mean())
rows.append(('ORACLE per-sample window', float(((orc_s - tru) ** 2).mean()), dir_acc(orc_s)))
rows.append(('ORACLE per-pair window (dir, loose bound)', float('nan'), orc_p_dir))

print(f"{'model':<44}{'corr MSE':>10}{'dir acc':>9}")
for n, m, a in rows: print(f'{n:<44}{m:>10.4f}{a:>9.3f}')
mr = [json.load(open(f))['corr_dir']['mean_reversion_rule'] for f in glob.glob(os.path.join(d, 'fixedw*_delta.json'))]
if mr: print(f"{'mean-reversion rule':<44}{'':>10}{mr[0]:>9.3f}")
print(f'\nbest single window: {best_single} bars')
print('per-sample oracle picks:', {w: int((pick_s == i).sum()) for i, w in enumerate(ws)})
gain = dir_acc(orc_s) - dir_acc(P[best_single])
print(f'per-sample oracle gain over best single window: {gain*100:+.2f} pp direction accuracy')
print('-> a learned chooser can gain at most this much (per-sample granularity).')
