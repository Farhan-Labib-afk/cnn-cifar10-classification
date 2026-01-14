"""
ETL script to generate BI-friendly metrics from results.json and
data/training_metrics.json. Outputs a normalized CSV for Power BI/Tableau.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def _parse_percent(value: str) -> float:
    value = value.strip()
    if value.endswith("%"):
        value = value[:-1]
    return float(value)


def extract_results(results_path: Path) -> dict:
    return json.loads(results_path.read_text(encoding="utf-8"))

def extract_training_metrics(metrics_path: Path) -> dict | None:
    if not metrics_path.exists():
        return None
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def transform_to_long(rows: dict, run_id: str) -> list[dict]:
    return [
        {
            "run_id": run_id,
            "model": "custom_cnn",
            "epoch": "",
            "metric": "best_accuracy_pct",
            "value": _parse_percent(rows["custom_best_acc"]),
            "unit": "percent",
        },
        {
            "run_id": run_id,
            "model": "custom_cnn",
            "epoch": "",
            "metric": "training_time_min",
            "value": float(rows["custom_time_min"]),
            "unit": "minutes",
        },
        {
            "run_id": run_id,
            "model": "resnet18_transfer",
            "epoch": "",
            "metric": "best_accuracy_pct",
            "value": _parse_percent(rows["resnet_best_acc"]),
            "unit": "percent",
        },
        {
            "run_id": run_id,
            "model": "resnet18_transfer",
            "epoch": "",
            "metric": "training_time_min",
            "value": float(rows["resnet_time_min"]),
            "unit": "minutes",
        },
    ]

def transform_epochs(metrics: dict, run_id: str) -> list[dict]:
    rows: list[dict] = []
    for model_name, history in metrics["models"].items():
        train_losses = history.get("train_losses", [])
        train_accs = history.get("train_accs", [])
        test_accs = history.get("test_accs", [])
        max_len = max(len(train_losses), len(train_accs), len(test_accs))
        for idx in range(max_len):
            epoch = idx + 1
            if idx < len(train_losses):
                rows.append(
                    {
                        "run_id": run_id,
                        "model": model_name,
                        "epoch": epoch,
                        "metric": "train_loss",
                        "value": float(train_losses[idx]),
                        "unit": "loss",
                    }
                )
            if idx < len(train_accs):
                rows.append(
                    {
                        "run_id": run_id,
                        "model": model_name,
                        "epoch": epoch,
                        "metric": "train_accuracy_pct",
                        "value": float(train_accs[idx]),
                        "unit": "percent",
                    }
                )
            if idx < len(test_accs):
                rows.append(
                    {
                        "run_id": run_id,
                        "model": model_name,
                        "epoch": epoch,
                        "metric": "test_accuracy_pct",
                        "value": float(test_accs[idx]),
                        "unit": "percent",
                    }
                )
    return rows


def load_to_csv(output_path: Path, rows: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["run_id", "model", "epoch", "metric", "value", "unit"]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate BI-ready metrics CSV from results.json."
    )
    parser.add_argument(
        "--results",
        default=str(BASE_DIR / "results.json"),
        help="Path to results.json",
    )
    parser.add_argument(
        "--metrics",
        default=str(BASE_DIR / "data" / "training_metrics.json"),
        help="Path to training_metrics.json",
    )
    parser.add_argument(
        "--out",
        default=str(BASE_DIR / "data" / "bi_metrics.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--run-id",
        default=default_run_id(),
        help="Run identifier for this export",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    output_path = Path(args.out)
    run_id = args.run_id

    raw = extract_results(results_path)
    rows = transform_to_long(raw, run_id)

    metrics_path = Path(args.metrics)
    training_metrics = extract_training_metrics(metrics_path)
    if training_metrics:
        run_id = training_metrics.get("run_id", run_id)
        rows.extend(transform_epochs(training_metrics, run_id))
    load_to_csv(output_path, rows)

    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
