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
- Logistic-regression baseline (2026-09-07): `taobao.features.counts` builds eleven
  count/recency features from the padded tensors (hours inverted from the log scale and
  rounded to whole seconds); `taobao.baseline` fits scaler + L2 logistic regression on
  train and scores one split. Validation AUC 0.585718, log loss 0.461970, bucket AUC
  0.549 / 0.586 / 0.598; last-24h cart count alone gives AUC 0.537289. The LSTM leads by
  +0.014 AUC on validation, inside the expected +0.005 to +0.02 range.

Immediate next work, following the plan in `CLAUDE.md`:

1. Run bounded LSTM tuning (at most four runs, learning rate and embedding dim only),
   then implement mean-pooling, GRU, and item embedding ablations behind one `--model`
   flag.
2. Record final baseline/model/ablation results and caveats in the README, scoring the
   test split exactly once.
3. Add a FastAPI endpoint with offline/online encoding parity, then measure latency and
   add Docker packaging.

## Next session: bounded LSTM tuning and ablations

Implement roadmap item 1 (CLAUDE.md Steps 5 and 6):

- Tuning: at most four `taobao.train` runs varying learning rate and embedding dim.
  Record each run's exact command and validation AUC; keep the best checkpoint.
- Ablations behind one `--model` flag: masked mean pooling (`(x * mask).sum(1) /
  lengths`, never divide by 50), GRU, and an item-embedding variant. Score each on
  validation with `taobao.score` and append rows to the single README table.
- Do not score the test split until Step 7. Do not push until the user says to.

## Code conventions

- Use Polars for dataframe work and Parquet for processed tabular artifacts.
- Keep modules small and focused; add tests for cutoff, labeling, vocabulary, padding,
  evaluation, and train/test isolation behavior.
- Reuse one encoding path for offline scoring and future serving.
- Use imperative commit messages.
- Mark unfinished features as planned rather than implying they exist.
