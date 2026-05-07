"""Analyze ablation sweep results from TensorBoard logs.
Ref: MATH.md M.3, CLAUDE.md Sec 8

Parses TensorBoard event files from runs/ and builds a summary table:
- R@k (centroid + cross-modal + img↔latex)
- Geometry (D_intra, D_inter, collapse)
- Ranking E1-E7

Usage:
    python code/analyze_ablation.py [--runs-dir runs/]
"""

import argparse
from pathlib import Path
from collections import defaultdict

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("ERROR: tensorboard not installed. Run: pip install tensorboard")
    raise


EXPERIMENTS = [
    "e1_pairwise", "e2_centroid", "e3_centroid_reg",
    "e4_composite_static", "e5_composite_learnable",
    "e6_fuzzy", "e7_lyapunov",
]

KEY_METRICS = [
    "metrics/centroid_R@1",
    "metrics/centroid_R@10",
    "metrics/img_to_latex_R@1",
    "metrics/img_to_en_R@1",
    "metrics/en_to_lean_R@1",
    "metrics/D_intra",
    "metrics/D_inter",
    "metrics/collapse_score",
    "metrics/modality_balance",
    "loss/total",
]


def find_run_dirs(runs_dir: Path) -> dict[str, Path]:
    """Find most recent run directory for each experiment."""
    exp_dirs = {}
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        for exp in EXPERIMENTS:
            if name.startswith(exp):
                exp_dirs[exp] = d  # last one wins (most recent)
    return exp_dirs


def extract_final_metrics(run_dir: Path) -> dict[str, float]:
    """Extract last value of key metrics from TensorBoard events."""
    ea = EventAccumulator(str(run_dir))
    ea.Reload()

    metrics = {}
    available_tags = ea.Tags().get("scalars", [])

    for tag in KEY_METRICS:
        if tag in available_tags:
            events = ea.Scalars(tag)
            if events:
                metrics[tag] = events[-1].value
        else:
            metrics[tag] = float("nan")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Analyze ablation sweep")
    parser.add_argument("--runs-dir", default="runs", help="TensorBoard runs directory")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        print(f"ERROR: {runs_dir} not found. Run experiments first.")
        return

    exp_dirs = find_run_dirs(runs_dir)
    if not exp_dirs:
        print(f"No experiment runs found in {runs_dir}")
        return

    print(f"Found {len(exp_dirs)} experiment runs:")
    for exp, d in exp_dirs.items():
        print(f"  {exp}: {d.name}")
    print()

    # Collect metrics
    all_metrics = {}
    for exp, d in exp_dirs.items():
        all_metrics[exp] = extract_final_metrics(d)

    # Print summary table
    short_names = {k: k.split("/")[-1] for k in KEY_METRICS}

    # Header
    header = f"{'Experiment':<28}"
    for tag in KEY_METRICS:
        header += f" {short_names[tag]:>14}"
    print(header)
    print("-" * len(header))

    # Rows
    for exp in EXPERIMENTS:
        if exp not in all_metrics:
            continue
        row = f"{exp:<28}"
        for tag in KEY_METRICS:
            val = all_metrics[exp].get(tag, float("nan"))
            row += f" {val:>14.4f}"
        print(row)

    print()

    # Ranking by centroid_R@1
    r1_tag = "metrics/centroid_R@1"
    ranked = sorted(
        [(exp, m.get(r1_tag, 0)) for exp, m in all_metrics.items()],
        key=lambda x: x[1],
        reverse=True,
    )
    print("Ranking by centroid_R@1:")
    for rank, (exp, val) in enumerate(ranked, 1):
        marker = " <-- BEST" if rank == 1 else ""
        print(f"  {rank}. {exp}: {val:.4f}{marker}")

    # Check hypotheses
    print("\nHypothesis checks:")
    e1 = all_metrics.get("e1_pairwise", {}).get(r1_tag, 0)
    e2 = all_metrics.get("e2_centroid", {}).get(r1_tag, 0)
    print(f"  T.2 (E2 > E1): {'PASS' if e2 > e1 else 'FAIL'} (E1={e1:.4f}, E2={e2:.4f})")

    img_tag = "metrics/img_to_latex_R@1"
    for exp in EXPERIMENTS:
        if exp in all_metrics:
            img_r1 = all_metrics[exp].get(img_tag, 0)
            n_est = 100  # rough estimate
            random_baseline = 1.0 / max(n_est, 1)
            status = "PASS" if img_r1 > random_baseline else "BELOW_RANDOM"
            print(f"  img→latex R@1 ({exp}): {img_r1:.4f} (random≈{random_baseline:.4f}) [{status}]")


if __name__ == "__main__":
    main()
