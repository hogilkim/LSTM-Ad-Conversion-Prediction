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

Historical **Polars** preparation run (2026-09-02, Apple M1 Pro): 14.9 s on the pipeline's
internal clock and about 6.5 GB peak RSS. These are historical Polars measurements; the
current implementation uses pandas/NumPy. The original cohort statistics were:

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

### pandas/NumPy implementation and benchmark

Preparation uses two chunked pandas CSV passes (1,000,000 rows per chunk): the first
identifies users with valid timestamps before seeded sampling; the second retains their
valid events. NumPy handles sampling and numeric transformations. PyArrow writes Parquet
and preserves typed sequence columns: int32 items/categories, int8 behaviors, and float32
hours. The retained events and per-cutoff histories still require memory; chunked CSV
reading does not make the entire pipeline streaming.

Full-data migration run (2026-10-27, Apple M1 Pro, 32 GiB RAM, Python 3.12.13,
pandas 3.1.0, NumPy 2.5.2, PyArrow 25.0.1), writing to a separate directory:

```bash
/usr/bin/time -l uv run --python 3.12 python -m taobao.data.prepare --n-users 200000 --seed 42 --out-dir data/migration-pandas-20261027/processed > data/migration-pandas-20261027/prepare.log 2> data/migration-pandas-20261027/prepare-time.txt
```

The parent directory was created before running the command. Use a fresh output directory
for another benchmark.

- Pipeline clock (`stats.json`, before output writing/readback): **64.0 s**.
- Entire command, including startup, output writing, and Parquet readback: **68.14 s**.
- Peak RSS from macOS `/usr/bin/time -l`: **10.95 GB** (10,949,902,336 bytes).

This single pandas run was slower and used more memory than the recorded Polars run.
The runs occurred on different dates; dependency versions and filesystem caches were
not controlled, so these are recorded measurements rather than a controlled comparison.

Migration verification matched both vocabularies, vocabulary sizes, and all preparation
statistics except command/timing against the existing artifacts. Train's 742,611 rows
and validation's 198,453 rows matched in their original order, including all Parquet
values and column types (pandas adds schema metadata). All six regenerated tensor arrays
matched the saved arrays byte for byte, including float32 normalization, lengths, dtypes,
and right padding. Six synthetic configurations also matched the original implementation
before Polars was removed. The tests cover legacy Parquet list/large-list columns and new
pandas Arrow-backed columns. The benchmark logs and verification report are under
`data/migration-pandas-20261027/` (gitignored).

The original data/model artifacts and recorded model metrics were preserved. No models
were retrained and no test predictions were scored during this migration.

### Build tensor artifacts

```bash
uv run python -m taobao.data.tensors --splits train val
```

The converter uses pandas/PyArrow Parquet filtering to load one requested split and only
the columns needed for conversion. It writes six right-padded `.npy` arrays per split
under `data/processed/tensors/`: int64 `items`, `cats`, `behs`, and `lengths`, plus float32
`hours` and `labels`. Hours use `log1p(hours) / log1p(168)` in float32. `load_split` opens
only the requested arrays as read-only memory maps. This command prepares train and
validation arrays; test preparation/evaluation remains separate from migration checks.

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
with the restored best checkpoint. Pass `--skip-test` to never open the test split, which
is the setting used while tuning on validation. Run `uv run python -m taobao.train --help`
for optional device, worker, optimization, and model-size settings.

### Tuning

Bounded to four runs, varying only the learning rate and the category embedding
dimension, with every other setting at its default (seed 42, batch size 256, hidden size
64, patience 3). Each run has its own checkpoint. Wall time is the trainer's own clock,
which excludes time the machine spends asleep; run 3 additionally took 49.9 min by the
outer clock because the Mac slept mid-run. Run 1's wall time was not recorded. Epoch-level
logs are in `models/tuning_logs/` (gitignored). Validation AUC moved by up to 0.005
between neighbouring epochs of the same run, so differences of that size between runs are
within noise.

| Run | Learning rate | Category dim | Best epoch | Stopped at | Val AUC | Val log loss | Wall time | Checkpoint |
|-----|---------------|--------------|------------|------------|---------|--------------|-----------|------------|
| 1 | 1e-3 | 32 | 2 | not recorded | 0.599836 | 0.461043 | not recorded | `models/conversion_lstm_best.pt` |
| 2 | 3e-4 | 32 | 5 | 8 | 0.599474 | 0.465478 | 39.2 min | `models/tune_lr3e-4_dim32.pt` |
| 3 | 1e-3 | 16 | 4 | 7 | 0.597117 | 0.459069 | 36.8 min | `models/tune_lr1e-3_dim16.pt` |
| 4 | 1e-3 | 64 | 3 | 6 | 0.595198 | 0.462535 | 30.1 min | `models/tune_lr1e-3_dim64.pt` |

Commands, in run order (run 1 is the default command above):

```bash
uv run python -m taobao.train --learning-rate 3e-4 --category-embedding-dim 32 --checkpoint models/tune_lr3e-4_dim32.pt --skip-test
uv run python -m taobao.train --learning-rate 1e-3 --category-embedding-dim 16 --checkpoint models/tune_lr1e-3_dim16.pt --skip-test
uv run python -m taobao.train --learning-rate 1e-3 --category-embedding-dim 64 --checkpoint models/tune_lr1e-3_dim64.pt --skip-test
```

The default configuration (run 1) stays the selected LSTM, and the LSTM row in the results
table uses `models/conversion_lstm_best.pt`. The smaller learning rate reached the same AUC
in five epochs instead of two. Halving the category dimension cost about 0.003 AUC, at the
edge of noise. Doubling it cost about 0.005 and overfit visibly: train loss kept falling
from epoch 3 while validation AUC and log loss got worse. None of the four runs was
scored on the test split.

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
