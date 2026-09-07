# CLAUDE.md

## Purpose

This repo exists to produce one honest, defensible resume bullet for summer 2027 MLE
internship applications (TikTok is a named target):

> Trained an LSTM recurrent neural network in PyTorch on N Taobao user behavior events
> with a time-based split, beat a logistic regression baseline by Z AUC, and served
> predictions at W ms with FastAPI in Docker.

Prior resume feedback: the work reads as software engineering because no model is ever
trained. The point here is a trained sequence model, an honest baseline comparison, and a
served endpoint. Everything else is scope creep.

Assume an interviewer will probe: LSTM gating and vanishing gradients, padding and
masking, time-based splits and leakage, and why an LSTM instead of a Transformer. Code
and README should make those answers easy to give.

## Problem

Given a Taobao user's last 50 behavior events, predict whether that user makes any
purchase in the next 24 hours.

This is conversion prediction, the user-level propensity question that sits underneath an
ads ranking system's bid. Frame it as conversion prediction everywhere: code, README,
commit messages, resume. Never call it recommendation. Do not claim it *is* ads ranking:
there is no candidate ad, no impression and no attribution window here, and an interviewer
who works on ranking will notice.

## Data

- Alibaba Taobao UserBehavior. Official source: Tianchi dataset 649. Downloaded from the
  Kaggle mirror `marwa80/userbehavior` using the `kaggle` CLI.
- Roughly 100M rows, 1M users, Nov 25 to Dec 3 2017, China time (UTC+8).
- No header row. Columns: `user_id, item_id, category_id, behavior_type, timestamp`.
  `behavior_type` is one of `pv, cart, fav, buy`. `timestamp` is Unix seconds.
- The raw file contains out-of-range junk timestamps; filter them to the Nov 25 to Dec 3
  window during preparation.
- Data is never committed. `data/` and `models/` are gitignored.

## Data preparation

Implemented in `src/taobao/data/prepare.py`.

- Sample 200K users with seed 42.
- Daily cutoffs at 00:00 UTC+8.
  - Train cutoffs: Nov 28, Nov 29, Nov 30, Dec 1 (`split = 0`)
  - Validation cutoff: Dec 2 (`split = 1`)
  - Test cutoff: Dec 3 (`split = 2`)
- Input is the last 50 events strictly before the cutoff. Minimum 3 events, otherwise the
  example is dropped.
- Label is 1 if any `buy` event falls in `[cutoff, cutoff + 24h)`.
- Vocabularies for item and category ids are built only from events before the validation
  cutoff. Ids seen fewer than 5 times map to index 1 (`rare`). Index 0 is padding.
- Behavior types map to 1..4 in the order `pv, cart, fav, buy`.
- Each event carries `hours_before_cutoff`, clipped at 168.
- Output: `data/processed/examples.parquet` with list columns `items, cats, behs, hours`
  plus `label, split, seq_len, user_id, cutoff`; vocabulary JSON files; `stats.json`.

## Model plan

Status as of 2026-09-07: Steps 1 to 5 are done; the baseline and LSTM validation rows
and the tuning table are in the README. Step 6 (ablations) is next. Ordered so that stopping after any step still
leaves a coherent project. Budget about 10 hours. Steps 1-5 produce the resume bullet,
6-7 produce the interview answers, 8-10 finish the sentence.

### Step 1 - Tensors (30-45 min)
Load `examples.parquet`, pre-pad once to right-padded numpy arrays (`items, cats, behs,
hours, lengths, labels`) per split and save as `.npy`. No `Dataset`, no `collate_fn`:
`max_len` is fixed at 50, so per-batch padding costs more than it saves. Feed
`log1p(hours) / log1p(168)`, never raw 0-168, or the hour feature dominates the
`N(0,1)`-initialised embeddings.
Done: `load_split(name)` returns tensors whose shapes and label means match `stats.json`.

### Step 2 - Evaluation harness (30-45 min)
`evaluate(y_true, y_prob)` returning AUC and log loss, plus AUC sliced by `seq_len`
bucket (3-10 / 11-30 / 31-50), plus a function that appends one row to the README table.
Written before any model so the baseline and the LSTM are scored by identical code.
Done: returns AUC about 0.50 on random predictions.

### Step 3 - Logistic regression baseline (60-90 min)
Count features over the same 50 events: `pv/cart/fav/buy` counts, distinct categories,
`seq_len`, hours since last event, hours since last cart-or-fav, events in last 24h,
cart count in last 24h, buy count in last 24h. Standardise, then sklearn
`LogisticRegression`. Cart adds are the strongest pre-purchase signal in this dataset, so
isolating them temporally is what makes the baseline strong. Also record the AUC of
"cart count in last 24h" alone as a sanity floor.
Done: validation AUC and log loss in the results table. This is the number to beat.

### Step 4 - LSTM (90-120 min)
`nn.Embedding` for categories (6,964) and behaviors (5) with `padding_idx=0`, concatenated
with log-hours. One `nn.LSTM(hidden_size=64)`, `pack_padded_sequence` with
`enforce_sorted=False`, final hidden state into a `Linear` head, explicit `nn.Dropout`
before the head. `BCEWithLogitsLoss`, Adam, early stopping on validation AUC.
No item embeddings in v1 (see Step 6c). No `pos_weight`: positives are 14.7%, which is not
rare, and it would decalibrate the sigmoid and break the log-loss column.
Two traps: `nn.LSTM(dropout=...)` is silently a no-op on a single layer, and the `h_n`
returned from an unpacked padded batch is the state at t=49, not at the last real step
(measured max abs difference 0.47 against the correct value). Packing makes `h_n` correct
by construction; `out[arange(B), lengths - 1]` on the unpacked output is exactly
equivalent and faster, and knowing why is a better interview answer than either.
Done: validation AUC and log loss next to the baseline. The resume bullet exists here.

### Step 5 - Bounded tuning (30-45 min, hard cap 4 runs)
Learning rate and embedding dim only. Stop at 4 runs whatever the outcome.
Done: best validation config recorded with the exact command that produced it.

### Step 6 - Ablations behind one `--model` flag (45-60 min)
  a. Mean pooling over embeddings. Must be masked: `(x * mask).sum(1) / lengths`. This is
     the load-bearing experiment, the only one that shows recurrence does real work, and
     the one most likely to be silently wrong. Padded `hours = 0` collides with "event at
     the cutoff", and an unmasked mean divides by 50 instead of `seq_len`.
  b. GRU. One-line swap.
  c. With item embeddings (552K x 32). Measures how much lift is item-level signal and
     bounds the fingerprint concern from repeated users. The median item appears 17 times
     in training and 84.5% appear under 50 times, so the expectation is that it barely
     moves; either result is a good README paragraph.
Cut: the Transformer encoder. Training is 5 minutes, making the comparison honest is 3
hours. Answer "why not a Transformer" in prose (fixed L=50 leaves no long-range
dependency for attention to solve, O(L^2) with more parameters and more regularisation
sensitivity at this data scale) plus the length-sliced AUC from Step 2.

### Step 7 - Read the test set once, write the README core (60-75 min)
Score baseline, LSTM and ablations on test. Add the `seq_len`-sliced AUC: if the LSTM's
edge over mean pooling grows with sequence length, that is direct evidence the recurrence
is doing order-dependent work. Write the problem, data, split, results table, and the
three honesty notes in "Known caveats" below.
Done: a project that is complete and defensible even if nothing below gets built.

### Step 8 - FastAPI endpoint (60 min)
POST raw events as JSON, return a purchase probability. Must reuse the Step 1 encode and
pad path exactly (sort order, last-50 truncation, vocab and rare mapping, behavior
mapping, cutoff-relative hour clipping) or training and serving will silently diverge.
Done: `curl` returns a probability matching the offline score for the same user.

### Step 9 - Latency and Docker (80-110 min)
200 sequential requests against local uvicorn, report p50 and p95. Measure locally, not
through the container: Docker Desktop's macOS VM adds network overhead that makes the
number both worse and less representative. Then `python:3.12-slim` with the CPU-only
torch wheel (`--index-url https://download.pytorch.org/whl/cpu`), single stage, about 1 GB
instead of 6.
Done: p50 and p95 in the README, `docker run` serves the identical app. The full bullet
exists here.

### Step 10 - README polish (45-60 min)
Model diagram and a short "questions I expect" section: LSTM gating and vanishing
gradients; why padding is harmless for a unidirectional last-state readout and exactly
when it is not (bidirectional, mean pooling, left padding); why an LSTM over a Transformer
at L=50; what the time split does and does not control for; why item embeddings were
dropped.

### Contingency, decided at Step 6a
If mean pooling ties the LSTM and the LSTM ties the baseline, that is a feature and data
ceiling, not an architecture problem. Stop tuning, report it honestly with the
length-sliced AUC, and spend the time on Steps 8-10. Ordering mean pooling before serving
is what surfaces this at hour 6 rather than hour 14.

## Known caveats

These are real properties of the dataset and the split. None of them is a bug, and every
one of them belongs in the README, because naming them is worth more in an interview than
any extra model.

1. **Train is weekdays, evaluation is the weekend.** In 2017 the train cutoffs Nov 28 to
   Dec 1 are Tue to Fri, the validation cutoff Dec 2 is a Saturday and the test cutoff
   Dec 3 is a Sunday. The positive rate shifts from 0.1473 to about 0.176. This is not
   leakage and the baseline-versus-model comparison survives it because both models eat
   the same shift, but log loss is systematically pessimistic while AUC, being rank-based,
   is unaffected. Do not add a day-of-week feature: there is no weekend in the training
   cutoffs to learn one from.
2. **Users overlap across splits.** 97.9% of test examples are users the model trained on,
   because one user cohort is sampled and every eligible user yields an example at every
   cutoff. This is temporally clean (all inputs precede their cutoff, all weights are fit
   before Dec 2) and it matches production, where a conversion model scores the same warm
   population daily. Never claim cold-start generalisation, never feed `user_id` to the
   model, and state that the 742,611 training rows are correlated repeated snapshots
   rather than independent users.
3. **The `hours` feature has different support in every split.** The mean age of a user's
   oldest retained event goes from 48.8h at the Nov 28 cutoff to 114.0h at Dec 3, and mean
   `seq_len` from 31.5 to 40.6, because the data window starts Nov 25. This is an artifact
   of the window, not of the world, and the model can in principle read the cutoff date
   off it.
4. **A vocabulary nit worth naming and dismissing.** Vocabularies are built from all events
   before Dec 2, which includes the label window of the last training cutoff, so a Nov 28
   item can be non-rare because it became frequent on Nov 29 to Dec 1. It is a global
   frequency aggregate over 200K users, per-example signal is negligible, and it is the
   standard construction. Do not change it; do name it.

## Evaluation rules

1. **Time-based splits only. Never a random split.** Any new feature or label must be
   computable from events strictly before the cutoff. If a change could see the future,
   it is wrong, even if the metric improves.
2. Always report the baseline next to the model. Never state a model number without its
   baseline and its split.
3. Metrics: AUC and log loss for models; p50 and p95 latency for serving.
4. Keep a single results table in the README. Every experiment updates that table rather
   than adding a new one.
5. Reproducibility: fixed seeds, a `stats.json` next to any processed artifact, and the
   exact command that produced each artifact recorded in the README.
6. Tune on validation. Read test numbers once, when results are final.
7. The LSTM is not required to win. If logistic regression beats it, that is the result:
   report it, and explain why in the README. Tuning until the model wins is how the whole
   table stops meaning anything.
8. Expect a small lift. Purchase propensity is strongly autocorrelated across days (test
   positive rate is 21.8% for users who bought on a training day against 14.8% for those
   who did not) and the baseline's buy-count feature already captures most of it. A
   realistic LSTM edge is +0.005 to +0.02 AUC, not +0.05.

## Repo layout

Planned; currently being scaffolded.

```
pyproject.toml          uv, Python 3.12, src layout, package `taobao`
src/taobao/data/        preparation (prepare.py)
src/taobao/features/    baseline count features
src/taobao/models/      PyTorch models (lstm, gru, mean pooling, transformer)
src/taobao/train.py     training loop and evaluation
src/taobao/serve.py     FastAPI app
scripts/                shell entry points (download, prepare, train, serve)
tests/                  pytest, focused on data logic
notebooks/              exploration only, never the source of truth
data/, models/          gitignored
```

Dependencies: polars, numpy, pyarrow, scikit-learn, torch, fastapi, uvicorn.
Dev: pytest, kaggle.

## Commands

Everything runs through `uv run`. Never `pip install` into the system Python.

```bash
uv sync                                   # install deps
uv run python -m taobao.data.prepare      # build data/processed/examples.parquet
uv run python -m taobao.train             # planned: train and evaluate
uv run uvicorn taobao.serve:app --reload  # planned: local serving
uv run pytest                             # tests
```

## Conventions

- Polars for data work, not Pandas. Parquet for processed data.
- Small files with one purpose. Tests for data logic, especially cutoffs, labeling, and
  vocabulary construction.
- Commit messages in imperative mood.
- Keep it simple: one architecture centered, one strong baseline, one ablation table, one
  served endpoint. No multimodal fusion, no feature explosion, no extra datasets.
- Do not invent numbers. Use `TBD` in the README until a run produces the value.
- Mark anything unbuilt as "planned" rather than describing it as done.

## Resume target

Describe results as conversion prediction from user behavior sequences. Fill in N, Z, and
W only from recorded runs, and always alongside the baseline and the split that produced
them.
