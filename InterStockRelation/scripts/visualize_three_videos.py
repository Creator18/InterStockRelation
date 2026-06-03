"""
Review artifact: render return / volume / volatility correlation videos side by
side for visual inspection before deletion + Trial 7.

3 snapshots (early / mid / late) x 3 videos. All Fiedler-ordered (identical
layout), so the same pixel = the same stock pair across all three.

Output: outputs/plots/three_videos_compare.png
"""
import os, pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA = os.path.join(ROOT, 'data', 'processed')
OUT  = os.path.join(ROOT, 'outputs', 'plots')

ret = np.load(os.path.join(DATA, 'corr_video.npy'))
vol = np.load(os.path.join(DATA, 'volume_corr_video.npy'))
rng = np.load(os.path.join(DATA, 'volat_corr_video.npy'))
with open(os.path.join(DATA, 'video_meta.pkl'), 'rb') as f:
    vm = pickle.load(f)
dates = vm['snap_dates']

T = ret.shape[0]
snaps = [int(T*0.1), int(T*0.5), int(T*0.9)]
videos = [('Return-corr', ret), ('Volume-corr', vol), ('Volatility-corr', rng)]

fig, axes = plt.subplots(len(snaps), 3, figsize=(15, 15))
fig.suptitle('Three Correlation Videos — same pixels (Fiedler order), different signals',
             fontsize=15, fontweight='bold')
norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

for r, s in enumerate(snaps):
    for c, (name, v) in enumerate(videos):
        ax = axes[r, c]
        im = ax.imshow(v[s], cmap='RdBu_r', norm=norm, interpolation='nearest', aspect='equal')
        if r == 0:
            ax.set_title(name, fontsize=12, fontweight='bold')
        if c == 0:
            ax.set_ylabel(f'{str(dates[s])[:10]}', fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
os.makedirs(OUT, exist_ok=True)
path = os.path.join(OUT, 'three_videos_compare.png')
plt.savefig(path, dpi=110, bbox_inches='tight')
plt.close()
print(f'Saved {path}')

# Also: how similar is the cluster (Fiedler-block) structure across videos?
# Correlate the three at a single frame to quantify visual difference.
off = ~np.eye(95, dtype=bool)
for s in snaps:
    a, b, c = ret[s][off], vol[s][off], rng[s][off]
    print(f'\nframe {s} ({str(dates[s])[:10]}) off-diagonal correlations:')
    print(f'  return  vs volume     : {np.corrcoef(a,b)[0,1]:+.3f}')
    print(f'  return  vs volatility : {np.corrcoef(a,c)[0,1]:+.3f}')
    print(f'  volume  vs volatility : {np.corrcoef(b,c)[0,1]:+.3f}')
