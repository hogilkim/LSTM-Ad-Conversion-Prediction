"""Score a saved checkpoint on one split with the shared metrics.

Example:
    uv run python -m taobao.score --split val --update-readme
"""

from __future__ import annotations

import argparse
from pathlib import Path

from taobao.data.dataset import TensorSplitDataset
from taobao.data.tensors import DEFAULT_OUT_DIR
from taobao.evaluation import EvaluationMetrics, update_readme_results
from taobao.model import ConversionLSTM
from taobao.train import (
    DEFAULT_CHECKPOINT,
    evaluate,
    load_checkpoint,
    make_data_loader,
    select_device,
)

DEFAULT_README = Path("README.md")


def score_checkpoint(
    checkpoint_path: Path,
    split: str,
    tensors_dir: Path = DEFAULT_OUT_DIR,
    batch_size: int = 256,
    device_name: str = "auto",
) -> EvaluationMetrics:
    """Rebuild the model from its checkpoint and evaluate it on ``split``."""
    device = select_device(device_name)
    checkpoint = load_checkpoint(checkpoint_path, device)
    model = ConversionLSTM(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    loader = make_data_loader(
        TensorSplitDataset(split, tensors_dir),
        batch_size,
        shuffle=False,
        device=device,
        seed=0,
        num_workers=0,
    )
    return evaluate(model, loader, device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--tensors-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--model-name", default="LSTM")
    parser.add_argument("--readme", type=Path, default=DEFAULT_README)
    parser.add_argument("--update-readme", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = score_checkpoint(
        args.checkpoint, args.split, args.tensors_dir, args.batch_size, args.device
    )
    split_label = "validation" if args.split == "val" else args.split
    print(f"split={split_label} auc={metrics.auc:.6f} log_loss={metrics.log_loss:.6f}")
    for bucket, value in metrics.auc_by_sequence_length.items():
        print(f"auc[seq_len {bucket}]={'N/A' if value is None else f'{value:.6f}'}")
    if args.update_readme:
        changed = update_readme_results(
            args.readme, model=args.model_name, split=split_label, metrics=metrics
        )
        print(f"readme_updated={changed}")


if __name__ == "__main__":
    main()
