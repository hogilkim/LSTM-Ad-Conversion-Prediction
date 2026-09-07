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
- Reproducible training, validation AUC/log-loss evaluation, early stopping, atomic best
  checkpoint saving, checkpoint restoration, and one-time test evaluation.

Immediate next work, following the plan in `CLAUDE.md`:

1. Finish the shared evaluation harness: add AUC by sequence-length bucket (3-10, 11-30,
   31-50) and a single results-table update path.
2. Implement the logistic-regression baseline on count and recency features, including
   the last-24-hour cart-count sanity floor.
3. Run bounded LSTM tuning (at most four runs), then implement mean-pooling, GRU, and item
   embedding ablations.
4. Record final baseline/model/ablation results and caveats in the README.
5. Add a FastAPI endpoint with offline/online encoding parity, then measure latency and
   add Docker packaging.

## Next session: shared evaluation metrics

Implement only the first roadmap item before starting the baseline:

- Create one model-agnostic evaluation path that accepts labels, probabilities, and
  sequence lengths so the future baseline and the LSTM use identical metric code.
- Return overall AUC and log loss, plus AUC for sequence-length buckets 3-10, 11-30, and
  31-50.
- Add one safe, deterministic path for updating the single README results table.
- Refactor the current LSTM evaluation code to use the shared metric calculation without
  changing its training behavior.
- Add focused tests, including random predictions near 0.50 AUC, bucket boundaries,
  invalid or empty inputs, and README table updates.
- Run the full test suite. Do not implement the logistic-regression baseline in this
  session and do not evaluate the real test split.
- Do not push to GitHub until the user explicitly says to push.
- After implementation, walk through every changed file chunk by chunk using concise,
  iTerm2-friendly plain text.

## Code conventions

- Use Polars for dataframe work and Parquet for processed tabular artifacts.
- Keep modules small and focused; add tests for cutoff, labeling, vocabulary, padding,
  evaluation, and train/test isolation behavior.
- Reuse one encoding path for offline scoring and future serving.
- Use imperative commit messages.
- Mark unfinished features as planned rather than implying they exist.
