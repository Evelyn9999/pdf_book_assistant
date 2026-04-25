import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_or_create_template(csv_path: Path) -> pd.DataFrame:
    if csv_path.exists():
        return pd.read_csv(csv_path)

    df = pd.DataFrame(
        [
            {"method": "TF-IDF baseline", "pass": 0, "partial": 0, "fail": 0, "refusal_correct": 0},
            {"method": "Final dual-channel", "pass": 0, "partial": 0, "fail": 0, "refusal_correct": 0},
        ]
    )
    df.to_csv(csv_path, index=False)
    print(f"Template created at: {csv_path}")
    print("Fill the numbers and run this script again.")
    return df


def plot_metrics(df: pd.DataFrame, output_path: Path):
    required_cols = ["method", "pass", "partial", "fail", "refusal_correct"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    methods = df["method"].tolist()
    categories = ["pass", "partial", "fail", "refusal_correct"]
    colors = ["#2ca02c", "#ffbf00", "#d62728", "#1f77b4"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = range(len(methods))
    width = 0.18

    for i, cat in enumerate(categories):
        offsets = [v + (i - 1.5) * width for v in x]
        bars = ax.bar(offsets, df[cat].tolist(), width=width, label=cat, color=colors[i])
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.05, f"{int(h)}", ha="center", va="bottom", fontsize=9)

    ax.set_title("QA Performance Comparison")
    ax.set_ylabel("Count")
    ax.set_xticks(list(x))
    ax.set_xticklabels(methods, rotation=0)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    print(f"Saved chart: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Pass/Partial/Fail/Refusal comparison chart.")
    parser.add_argument("--csv", default="evaluation_summary.csv", help="Input CSV path")
    parser.add_argument("--out", default="comparison_chart.png", help="Output image path")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out)
    df = load_or_create_template(csv_path)
    plot_metrics(df, out_path)
