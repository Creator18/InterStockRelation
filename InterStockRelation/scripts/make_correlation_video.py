"""
Generate visual artifacts of the correlation video so you can see what
the model is actually being trained on.

Outputs (saved to outputs/plots/):
  1. trial_video_fiedler.gif   - main video: Fiedler-reordered correlation matrix
                                  evolving frame-by-frame. This is what the
                                  CNN/ConvLSTM actually "sees".
  2. trial_video_unordered.gif  - same video but stocks in original order, so
                                  you can compare and see why Fiedler reordering
                                  matters.
  3. trial_channels_snapshot.png - one timestep showing all 5 input channels
                                   side-by-side (raw, d1, d5, roll_mean, roll_std)
  4. trial_video_mp4.mp4        - MP4 if ffmpeg is available (better quality)

Run from project root:  python scripts/make_correlation_video.py
"""
import os, pickle, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import TwoSlopeNorm

# ── Config ────────────────────────────────────────────────────────────────────
N_FRAMES_TO_SHOW = 200      # ~13 sec at 15 fps, manageable file size
FPS              = 15
STRIDE           = 1        # use every Nth frame (1 = no skip)
START_FRAME      = 100      # skip warmup window — first frames have less data

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA = os.path.join(ROOT, 'data', 'processed')
OUT  = os.path.join(ROOT, 'outputs', 'plots')
os.makedirs(OUT, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
print('Loading correlation video...')
corr_video  = np.load(os.path.join(DATA, 'corr_video.npy'))
fiedler_ord = np.load(os.path.join(DATA, 'fiedler_order.npy'))
snap_ts     = np.load(os.path.join(DATA, 'snap_timesteps.npy'), allow_pickle=True)
with open(os.path.join(DATA, 'pair_meta.pkl'), 'rb') as f:
    pair_meta = pickle.load(f)
tickers = pair_meta['tickers']

T, N, _ = corr_video.shape
print(f'  Shape: {corr_video.shape}  ({T} snapshots, {N} stocks)')
print(f'  Range: [{corr_video.min():.3f}, {corr_video.max():.3f}]')

# Apply Fiedler reordering so cluster structure is visually apparent
fiedler_video = corr_video[:, fiedler_ord, :][:, :, fiedler_ord]
print(f'  Fiedler-reordered: stocks grouped by cluster structure')

# Subset frames
end_frame = min(START_FRAME + N_FRAMES_TO_SHOW * STRIDE, T)
frame_idx = list(range(START_FRAME, end_frame, STRIDE))
print(f'  Animating frames [{frame_idx[0]} .. {frame_idx[-1]}]  ({len(frame_idx)} frames)')

# ── Helper: animate a (T,N,N) array to GIF ────────────────────────────────────
def animate_video(video, frame_indices, title, out_path, fps=15, cmap='RdBu_r'):
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor('white')

    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    im = ax.imshow(video[frame_indices[0]], cmap=cmap, norm=norm,
                   interpolation='nearest', aspect='equal')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Correlation', fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    title_text = ax.set_title(f'{title}\nframe {frame_indices[0]}',
                              fontsize=12, fontweight='bold')

    def update(i):
        f = frame_indices[i]
        im.set_data(video[f])
        title_text.set_text(f'{title}\nframe {f}  '
                            f'(snapshot {snap_ts[f] if f < len(snap_ts) else f})')
        return [im, title_text]

    anim = animation.FuncAnimation(fig, update, frames=len(frame_indices),
                                   interval=1000 // fps, blit=False)

    print(f'  Saving {out_path} ...')
    anim.save(out_path, writer='pillow', fps=fps, dpi=80)
    plt.close(fig)
    print(f'  Done.  Size: {os.path.getsize(out_path) / 1e6:.1f} MB')

# ── 1. Fiedler-ordered correlation video ──────────────────────────────────────
print('\n[1/3] Generating Fiedler-ordered video...')
animate_video(
    fiedler_video, frame_idx,
    'Correlation Matrix Video (Fiedler-reordered)\nstocks grouped by cluster — what the CNN sees',
    os.path.join(OUT, 'trial_video_fiedler.gif'),
    fps=FPS
)

# ── 2. Unordered correlation video ────────────────────────────────────────────
print('\n[2/3] Generating unordered video (alphabetical stock order)...')
animate_video(
    corr_video, frame_idx,
    'Correlation Matrix Video (alphabetical order)\nno cluster structure visible',
    os.path.join(OUT, 'trial_video_unordered.gif'),
    fps=FPS
)

# ── 3. Multi-channel snapshot showing what trial5+ models see ─────────────────
print('\n[3/3] Generating multi-channel snapshot...')

def build_multichannel(corr_video, delta_short=1, delta_long=5, win=4):
    T, H, W = corr_video.shape
    C = 5
    out = np.zeros((T, C, H, W), dtype=np.float32)
    out[:, 0] = corr_video
    out[delta_short:, 1] = corr_video[delta_short:] - corr_video[:-delta_short]
    out[delta_long:,  2] = corr_video[delta_long:]  - corr_video[:-delta_long]
    cum, cum2 = np.cumsum(corr_video, 0), np.cumsum(corr_video**2, 0)
    for t in range(T):
        lo = max(0, t - win + 1)
        n = t - lo + 1
        s  = cum[t]  - (cum[lo-1]  if lo > 0 else 0.0)
        s2 = cum2[t] - (cum2[lo-1] if lo > 0 else 0.0)
        mean = s / n
        var = np.maximum(0.0, s2 / n - mean**2)
        out[t, 3] = mean
        out[t, 4] = np.sqrt(var)
    return out

mc = build_multichannel(corr_video)
mc_fiedler = mc[:, :, fiedler_ord, :][:, :, :, fiedler_ord]

snapshot_t = frame_idx[len(frame_idx) // 2]  # middle frame
channels = mc_fiedler[snapshot_t]            # (5, 95, 95)
names    = ['raw correlation', '1-step delta', '5-step delta',
            '4-step rolling mean', '4-step rolling std']
ranges   = [(-1, 1), (-0.3, 0.3), (-0.5, 0.5), (-1, 1), (0, 0.4)]
cmaps    = ['RdBu_r', 'RdBu_r', 'RdBu_r', 'RdBu_r', 'viridis']

fig, axes = plt.subplots(1, 5, figsize=(22, 5))
fig.suptitle(f'5-channel input at snapshot t={snapshot_t}\n'
             f'(what the multi-channel models see per timestep)',
             fontsize=13, fontweight='bold')

for i, (ax, name, (vmin, vmax), cmap) in enumerate(zip(axes, names, ranges, cmaps)):
    if vmin < 0:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        im = ax.imshow(channels[i], cmap=cmap, norm=norm,
                       interpolation='nearest', aspect='equal')
    else:
        im = ax.imshow(channels[i], cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation='nearest', aspect='equal')
    ax.set_title(f'channel {i}: {name}', fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
out_png = os.path.join(OUT, 'trial_channels_snapshot.png')
plt.savefig(out_png, dpi=120, bbox_inches='tight')
plt.close(fig)
print(f'  Saved {out_png}')

# ── 4. Try MP4 if ffmpeg is available ─────────────────────────────────────────
print('\n[bonus] Trying to make an MP4 (better quality, smaller file)...')
try:
    if animation.FFMpegWriter.isAvailable():
        fig, ax = plt.subplots(figsize=(7, 7))
        norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
        im = ax.imshow(fiedler_video[frame_idx[0]], cmap='RdBu_r', norm=norm,
                       interpolation='nearest', aspect='equal')
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks([]); ax.set_yticks([])
        title_text = ax.set_title(f'frame {frame_idx[0]}', fontsize=12)

        def update(i):
            f = frame_idx[i]
            im.set_data(fiedler_video[f])
            title_text.set_text(f'frame {f}')
            return [im, title_text]

        anim = animation.FuncAnimation(fig, update, frames=len(frame_idx),
                                       interval=1000 // FPS, blit=False)
        mp4_path = os.path.join(OUT, 'trial_video_mp4.mp4')
        anim.save(mp4_path, writer='ffmpeg', fps=FPS, dpi=100)
        plt.close(fig)
        print(f'  Saved {mp4_path}  ({os.path.getsize(mp4_path) / 1e6:.1f} MB)')
    else:
        print('  ffmpeg not available — skipping MP4. GIFs are good enough.')
except Exception as e:
    print(f'  MP4 generation skipped: {e}')

print('\nAll done. Open these in your file explorer:')
for f in ['trial_video_fiedler.gif', 'trial_video_unordered.gif',
          'trial_channels_snapshot.png', 'trial_video_mp4.mp4']:
    p = os.path.join(OUT, f)
    if os.path.exists(p):
        print(f'  {p}')
