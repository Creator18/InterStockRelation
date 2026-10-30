# InterStockRelation

Forecasting how the correlation structure of a stock market changes, by treating the
full stock-by-stock correlation matrix as a video.

## The idea

At each time step, the correlations between 95 Nifty 100 stocks form a 95 x 95 matrix.
Stocks are reordered with the Fiedler vector of the correlation graph, so stocks that move
together sit next to each other and sectors appear as blocks. A sequence of these matrices
is then a video, and the task becomes video prediction: given the last 20 daily frames,
predict the frame 5 trading days ahead.

The project then asks a harder question: how much of this is actually predictable, and
does any model, input or resolution beat simple baselines?

## Data

5-minute OHLCV bars for 95 Nifty 100 constituents, July 30, 2021 to April 8, 2026
(87,122 aligned timesteps). The raw market data is **not included** in this repository.
Small derived metadata is in `data/processed/`. External series used in the
augmentation tests are in `data/external/`: the Geopolitical Risk index (Caldara and
Iacoviello) and GDELT sector news tone.

## Main results (daily resolution)

These numbers are from the manuscript source in `paper/samplepaper.tex`.

- **A shared accuracy ceiling.** Five video architectures, trained with the same combined loss
  and multi-channel inputs, all land at about 73% directional accuracy and MSE between 0.0681
  and 0.0721. The strongest baseline, predicting the mean frame, has MSE 0.071. Changing the
  architecture or adding volume or volatility correlation does not move this ceiling.
- **The ceiling is mostly mean reversion.** A one-line rule that predicts correlations
  revert toward their 20-snapshot mean already reaches about 73%.
- **Accuracy depends on how big the change is.** Directional accuracy rises from 73% over
  all pairs to 88% on correlation changes larger than 0.30.
- **Selective prediction.** If we keep only the calls where the model predicts a large
  change, directional accuracy goes from 73.7% at full coverage to 85.4% at 50% coverage,
  91.5% at 25% and 95.6% at the top 10%. Predicted and actual change magnitudes correlate
  at 0.47.
- **Cross-stock inference.** Used to infer one stock's move from a peer's, the forecast
  correlation scores 0.283. That beats the historical mean (0.276) and persistence (0.188).
- **External data adds nothing.** India VIX, geopolitical risk and news tone add no skill
  beyond what the correlation matrix already contains.

Caveat: the selective-prediction gains come largely from mean reversion on large moves.
The model's edge over the simple mean-reversion rule is small.

## Intraday follow-up (5-minute resolution, work in progress)

These results come from the JSON files in `outputs/trial10/` and `outputs/trial11/`.
They are not part of the manuscript.

- **Trial 10, looped transformer.** It works on raw 5-minute stock tokens, learns its own
  time windows on each pass, learns when to stop looping, and attends across stocks. The
  target is the correlation over the next 2 hours. It does **not** beat simple baselines:
  - Directional accuracy is 0.701, against 0.723 for the mean-reversion rule.
  - On the 25% of calls where it predicts the largest change, accuracy is 0.871, against
    0.901 for the mean-reversion rule's own top 25%.
  - Correlation MSE is 0.0721, against 0.0582 for historical correlation.
  - A fixed window schedule performs the same as learned windows (0.702).
- **Trial 11, which stocks does a stock depend on?** The model hides one stock's current
  5-minute return and predicts it from the other 94 stocks. Its cross-attention then shows
  which stocks it used.
  - The top-attended stocks are the real sector peers. For example, CANBK attends to
    BANKBARODA, SBIN and PNB, and TCS attends to INFY, TECHM and WIPRO.
  - Across the two seeds, 54.5% and 59.3% of each stock's top-5 attended stocks are also
    among its top-5 by correlation.
  - Prediction R^2 is 0.291 and 0.302 for the two seeds. A linear ridge regression
    reaches 0.313.
  - The attention map is static. It does not follow correlation changes over the test
    period.

## Repository structure

```
notebooks/        trial1-9 (model trials), analysis notebooks (ablation, magnitude,
                  onset, groups, news, systemic risk, intraday probes)
scripts/          build_*.py generate the analysis notebooks; test_*.py validation tests;
                  trial10.py / trial11.py intraday models; *.sbatch / submit_*.sh for Slurm
data/processed/   small derived metadata (Fiedler order, timestamps, tickers)
data/external/    GPR index and GDELT tone series
models/           training histories (*.pkl); weights are not tracked
outputs/plots/    figures; outputs/trial10, trial11: intraday result JSONs and attention maps
paper/            manuscript source and figures
trial1_archive/   first, pair-level experiments (XGBoost, LSTM, temporal GNN)
```

## Setup

Python 3.11. Install the dependencies:

```bash
pip install -r requirements.txt
```

Use a CUDA build of PyTorch for training. On V100 GPUs, use the CUDA 12.x builds; CUDA 13
builds no longer support them.

## Reproducing

1. Place the raw 5-minute bar files in `data/raw/` as `<TICKER>_5minute.csv`, with the
   columns `date, open, high, low, close, volume`. The index series go in the same folder:
   `NIFTY 50_5minute.csv` and `INDIA VIX_5minute.csv`.
2. Daily pipeline. `notebooks/trial1.ipynb` and `notebooks/trial2.ipynb` build the pair
   cache and `data/processed/corr_video.npy` (EWMA correlation, Fiedler order, daily
   snapshots). Then run `scripts/extract_aux_videos.py` and `scripts/build_volvol_videos.py`
   for the auxiliary videos.
3. Trials and analyses. Each analysis notebook is generated by a builder script, for example:
   ```bash
   python scripts/build_trial8.py
   ```
   Then run the generated notebook. The validation tests run directly as
   `python scripts/test_<name>.py`.
4. Intraday models:
   ```bash
   python scripts/prep_trial10_data.py
   python scripts/trial10.py --mode learned --halting 1
   python scripts/trial11.py --seed 0
   ```
   Add `--smoke` for a quick CPU check. On a Slurm cluster, use `scripts/submit_trial10.sh`
   or `scripts/submit_trial11.sh`. Runs checkpoint and resume, and can move to a better GPU
   when one frees up (`scripts/gpu_lease.py`).

## Status

A manuscript describing the daily-resolution study is under review. The intraday work is
ongoing.
