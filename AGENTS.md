# Repository instructions

## Goal

Build an honest, reproducible conversion-prediction project from Taobao user behavior:
compare a sequence LSTM with a strong logistic-regression baseline, then serve the chosen
model through FastAPI. Do not describe this project as recommendation or ads ranking.

## Tooling and commands

- Use Python 3.12 and `uv`; do not install packages into the system Python.
- Run tests with `uv run pytest`.
- Build processed examples with `uv run python -m taobao.data.prepare`.
- Build tensor artifacts with `uv run python -m taobao.data.tensors`.
- Train the LSTM with `uv run python -m taobao.train`.
- Score a saved checkpoint with `uv run python -m taobao.score --split val --update-readme`.
- Keep terminal output and handoffs concise and iTerm2-friendly.

## Data and evaluation invariants

- Use the existing time-based train/validation/test splits. Never replace them with a
  random split.
- Inputs for an example may only use events strictly before its cutoff.
- Vocabulary building must not see validation or test days. Index 0 is padding and index
  1 is the rare/unknown value.
- Preserve right padding and real sequence lengths. The LSTM must ignore padded steps.
- Tune and select checkpoints on validation data only. Evaluate test only after model
  selection is finished.
- Report AUC and log loss for every model, always beside the logistic-regression baseline
  on the same split.
- Do not invent or selectively omit results. The LSTM is not required to beat the
  baseline.
- Do not commit raw/processed data, checkpoints, or other generated model artifacts.

## Implementation status

Completed:

- Raw-data preparation and time-based examples.
- Fixed, memory-mapped NumPy tensor artifacts.
- `TensorSplitDataset`.
- Length-aware `ConversionLSTM` using packed sequences.
- Reproducible training, early stopping, atomic best checkpoint saving, checkpoint
  restoration, and one-time test evaluation.
- Shared evaluation in `taobao.evaluation`: `evaluate(labels, probabilities,
  sequence_lengths)` returns AUC, log loss, and AUC per sequence-length bucket (3-10,
  11-30, 31-50); `update_readme_results` atomically upserts one row in the single marked
  README results table. The trainer uses this path.
- `taobao.score`: rebuild a model from a checkpoint and score one split with the shared
  metrics, optionally writing the README row.
- Recorded LSTM validation row (2026-09-07) from the seed-42 checkpoint
  `models/conversion_lstm_best.pt`, epoch 2: AUC 0.599836, log loss 0.461043, bucket AUC
  0.585 / 0.599 / 0.605. The test split has not been scored through the README path.

Immediate next work, following the plan in `CLAUDE.md`:

1. Implement the logistic-regression baseline on count and recency features, including
   the last-24-hour cart-count sanity floor. This is the number the LSTM must be compared
   against; the LSTM row alone means nothing.
2. Run bounded LSTM tuning (at most four runs), then implement mean-pooling, GRU, and item
   embedding ablations.
3. Record final baseline/model/ablation results and caveats in the README.
4. Add a FastAPI endpoint with offline/online encoding parity, then measure latency and
   add Docker packaging.

## Next session: logistic-regression baseline

Implement only roadmap item 1 (CLAUDE.md Step 3):

- Build count features from the same right-padded tensors the LSTM reads
  (`data/processed/tensors/{split}_*.npy`, loaded through `taobao.data.tensors.load_split`),
  so both models see exactly the same 50 events. Note the `hours` array is already
  transformed to `log1p(hours) / log1p(168)`; invert it or read `examples.parquet` for raw
  hours, and state which in the code.
- Features: `pv/cart/fav/buy` counts, distinct categories, `seq_len`, hours since last
  event, hours since last cart-or-fav, events in last 24h, cart count in last 24h, buy
  count in last 24h. Padded positions must not count.
- Standardise, then sklearn `LogisticRegression`. Fit on train only.
- Score validation with `taobao.evaluation.evaluate` and write the row with
  `update_readme_results(model="Logistic regression baseline", split="validation")`.
- Also record the AUC of "cart count in last 24h" alone as a sanity floor, in prose next
  to the table, not as a table row.
- Tests: feature counts on a small hand-built padded batch, padding ignored, last-24h
  windows at the boundary.
- Record the exact command in the README. Do not score the test split. Do not tune the
  LSTM. Do not push until the user says to.
- After implementation, walk through the changed files one chunk at a time, concisely,
  checking the user's understanding per chunk. The user has a CS background and is new to
  ML.

## Code conventions

- Use Polars for dataframe work and Parquet for processed tabular artifacts.
- Keep modules small and focused; add tests for cutoff, labeling, vocabulary, padding,
  evaluation, and train/test isolation behavior.
- Reuse one encoding path for offline scoring and future serving.
- Use imperative commit messages.
- Mark unfinished features as planned rather than implying they exist.
