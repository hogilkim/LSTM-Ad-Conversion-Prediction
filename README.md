# LSTM-Ad-Conversion-Prediction

Ad conversion-rate prediction (pCVR) from user behavior sequences. Given a Taobao user's last 50
behavior events (page views, cart adds, favorites, purchases), predict whether that user
makes any purchase in the next 24 hours. An LSTM in PyTorch is compared against a
logistic regression baseline on count features, using strictly time-based splits
(daily cutoffs, no random splits), and the model is served through a FastAPI endpoint
(planned). Data: Alibaba Taobao UserBehavior, official source
[Tianchi dataset 649](https://tianchi.aliyun.com/dataset/649), downloaded from the Kaggle
mirror `marwa80/userbehavior`.

**Scope.** This is the user-level conversion propensity that sits underneath an ads bid,
where `bid = value x pCVR`. It is trained on organic e-commerce behavior: the dataset
contains no ad impressions, no candidate ads and no attribution window, so this is a
conversion model, not an ads ranking system, and is not claimed to be one.

## Setup

```bash
uv sync
```

## Download the data

Requires Kaggle credentials at `~/.kaggle/kaggle.json`. Writes `data/raw/UserBehavior.csv`
(about 3.7 GB, no header row; columns `user_id, item_id, category_id, behavior_type, timestamp`).

```bash
bash scripts/download_data.sh
```

## Prepare the dataset

Filters junk timestamps to the Nov 25 to Dec 3 2017 window (UTC+8), samples 200K users
with seed 42, builds (user, daily cutoff) examples from the last 50 events before each
cutoff, labels each with whether a `buy` happens in the following 24 hours, and writes
`data/processed/examples.parquet`, vocabulary JSON files, and `stats.json`.

```bash
uv run python -m taobao.data.prepare --n-users 200000 --seed 42
```

Recorded run (2026-09-02, Apple M1 Pro, 14.9 s wall, about 6.5 GB peak RSS):

| | value |
|---|---|
| Raw rows | 100,150,807 |
| Dropped out-of-window rows (junk timestamps) | 55,576 |
| Users with valid events / sampled | 987,991 / 200,000 (seed 42) |
| Sampled events | 20,254,681 |
| Vocabulary sizes (incl. pad and rare) | items 552,305, categories 6,964, behaviors 5 |
| Train (cutoffs Nov 28 to Dec 1, split 0) | 742,611 examples, positive rate 0.1473, mean seq_len 31.5 |
| Validation (cutoff Dec 2, split 1) | 198,453 examples, positive rate 0.1759, mean seq_len 37.6 |
| Test (cutoff Dec 3, split 2) | 199,863 examples, positive rate 0.1763, mean seq_len 40.6 |

Vocabularies are built only from events before the validation cutoff (Dec 2), so item and
category indices never see validation or test days; ids seen fewer than 5 times map to the
`rare` index 1 and index 0 is padding. Full details live in `data/processed/stats.json`.

## Tests

```bash
uv run pytest
```

## Train and evaluate the LSTM

After building the tensor artifacts, run the reproducible training pipeline:

```bash
uv run python -m taobao.train
```

It uses seed 42, batch size 256, Adam, and at most 15 epochs by default. Training data is
shuffled; validation and test data are not. Validation AUC selects the checkpoint in
`models/conversion_lstm_best.pt`, with early stopping after three epochs without an AUC
improvement. The test split is loaded only after training, and is evaluated exactly once
with the restored best checkpoint. Run `uv run python -m taobao.train --help` for optional
device, worker, optimization, and model-size settings.

## Logistic regression baseline

The baseline reads the same right-padded tensors as the LSTM, so both models see exactly
the same 50 events per example. It builds eleven count and recency features (`pv`, `cart`,
`fav`, `buy` counts, distinct categories, `seq_len`, hours since the last event, hours
since the last cart-or-fav, and events, carts and buys in the last 24 hours), standardises
them, and fits an L2 logistic regression on the train split only:

```bash
uv run python -m taobao.baseline --split val --update-readme
```

Recorded run (2026-09-07): features built in 2.0 s, fit in 0.3 s, model saved to
`models/baseline_logreg.joblib`. The largest standardised coefficients were `buy_count`
(+0.246), `events_last_24h` (+0.171), `hours_since_last_event` (-0.115) and
`hours_since_last_cart_or_fav` (-0.111). As a sanity floor, the last-24h cart count used
directly as a score, with no model at all, gives validation AUC 0.537289.

## Results

Rows are written by the scoring commands, never by hand. The baseline row came from the
command above. The LSTM validation row came from the seed-42 checkpoint produced by
`uv run python -m taobao.train`, scored with:

```bash
uv run python -m taobao.score --split val --update-readme
```

<!-- results-table:start -->
| Model | Split | AUC | Log loss | AUC (seq_len 3-10) | AUC (seq_len 11-30) | AUC (seq_len 31-50) |
|-------|-------|-----|----------|--------------------|---------------------|---------------------|
| Logistic regression baseline | validation | 0.585718 | 0.461970 | 0.548767 | 0.585529 | 0.598380 |
| LSTM | validation | 0.599836 | 0.461043 | 0.585252 | 0.599446 | 0.604940 |
<!-- results-table:end -->
