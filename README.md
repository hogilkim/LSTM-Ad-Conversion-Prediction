# taobao-conversion-lstm

Conversion prediction from user behavior sequences. Given a Taobao user's last 50
behavior events (page views, cart adds, favorites, purchases), predict whether that user
makes any purchase in the next 24 hours. An LSTM in PyTorch is compared against a
logistic regression baseline on count features, using strictly time-based splits
(daily cutoffs, no random splits), and the model is served through a FastAPI endpoint
(planned). Data: Alibaba Taobao UserBehavior, official source
[Tianchi dataset 649](https://tianchi.aliyun.com/dataset/649), downloaded from the Kaggle
mirror `marwa80/userbehavior`.

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

## Tests

```bash
uv run pytest
```

## Results

| Model | Split | AUC | Log loss |
|-------|-------|-----|----------|
| Logistic regression baseline | TBD | TBD | TBD |
| LSTM | TBD | TBD | TBD |
