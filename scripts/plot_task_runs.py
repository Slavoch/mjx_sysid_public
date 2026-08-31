from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
for path in (REPO_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot checkpoint playback for every dataset run in a task."
    )
    parser.add_argument("--task", required=True, help="Task key, e.g. encos8112.pd_then_changers.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Checkpoint .pkl path.")
    return parser.parse_args()


def resolve_checkpoint(path: Path) -> Path:
    checkpoint = path if path.is_absolute() else (REPO_ROOT / path).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    return checkpoint


def run_dir_for(checkpoint: Path) -> Path:
    return checkpoint.parent.parent if checkpoint.parent.name == "checkpoints" else checkpoint.parent


KIND_COLORS = {
    "chirp": "tab:blue",
    "sine": "tab:orange",
    "steps": "tab:green",
    "linear": "tab:purple",
    "hold": "tab:gray",
    "other": "tab:brown",
}


def rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values**2)))


def motion_kind(stem: str) -> str:
    name = stem.lower()
    for kind in ("chirp", "linear", "sine", "steps", "hold"):
        if kind in name:
            return kind
    return "other"


def short_label(run_index: int, stem: str) -> str:
    name = stem.removeprefix("pd_")
    return f"{run_index:02d}  {name}"


def save_run_figure(
    *,
    figure_path: Path,
    title: str,
    dt: float,
    sim_qpos: np.ndarray,
    gt_qpos: np.ndarray,
    qact: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    err = sim_qpos - gt_qpos
    t = np.arange(len(err)) * dt

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle(title)

    axes[0].plot(t, gt_qpos, label="dataset qpos", linewidth=1.5)
    axes[0].plot(t, sim_qpos, label="sim qpos", linewidth=1.0)
    axes[0].plot(t, qact, label="dataset qact", color="tab:green", linewidth=1.0)
    axes[0].set_ylabel("qpos / qact [rad]")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(t, err, label="sim - dataset", color="tab:red", linewidth=1.0)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("error [rad]")
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)


def save_summary_figure(*, metrics: pd.DataFrame, figure_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    df = metrics.copy()
    df["kind"] = [motion_kind(Path(path).stem) for path in df["file"]]
    df["label"] = [
        short_label(int(run_index), Path(path).stem)
        for run_index, path in zip(df["run_index"], df["file"])
    ]
    df = df.sort_values("rmse_rad", ascending=True)

    mean_rmse = float(df["rmse_rad"].mean())
    mean_mae = float(df["mae_rad"].mean())
    mean_max = float(df["max_abs_rad"].mean())
    best = df.iloc[0]
    worst = df.iloc[-1]
    kind_order = [kind for kind in KIND_COLORS if kind in set(df["kind"])]
    family_rmse = df.groupby("kind")["rmse_rad"].mean().reindex(kind_order)

    n_runs = len(df)
    fig = plt.figure(figsize=(14, max(7.5, 0.34 * n_runs + 3.0)), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, max(2.4, 0.13 * n_runs)])
    ax_overall = fig.add_subplot(grid[0, 0])
    ax_family = fig.add_subplot(grid[0, 1])
    ax_runs = fig.add_subplot(grid[1, :])

    overall_vals = [mean_rmse, mean_mae, mean_max]
    overall_bars = ax_overall.bar(
        ["RMSE", "MAE", "max |err|"],
        overall_vals,
        color=["tab:red", "tab:blue", "tab:purple"],
        alpha=0.85,
    )
    ax_overall.set_ylabel("rad")
    ax_overall.set_title("Mean over all runs")
    ax_overall.grid(True, axis="y", alpha=0.3)
    ax_overall.bar_label(overall_bars, fmt="%.4f", padding=3)
    ax_overall.set_ylim(0.0, max(overall_vals) * 1.25)

    family_bars = ax_family.bar(
        family_rmse.index,
        family_rmse.values,
        color=[KIND_COLORS[kind] for kind in family_rmse.index],
        alpha=0.85,
    )
    ax_family.set_ylabel("mean RMSE [rad]")
    ax_family.set_title("Mean RMSE by motion type")
    ax_family.grid(True, axis="y", alpha=0.3)
    ax_family.bar_label(family_bars, fmt="%.4f", padding=3)

    run_colors = [KIND_COLORS[kind] for kind in df["kind"]]
    ax_runs.barh(df["label"], df["rmse_rad"], color=run_colors, alpha=0.85)
    ax_runs.axvline(mean_rmse, color="black", linestyle="--", linewidth=1.0)
    ax_runs.set_xlabel("RMSE [rad]")
    ax_runs.set_title("RMSE per run")
    ax_runs.grid(True, axis="x", alpha=0.3)
    ax_runs.tick_params(axis="y", labelsize=8)
    ax_runs.legend(
        handles=[Patch(color=KIND_COLORS[kind], label=kind) for kind in kind_order]
        + [plt.Line2D([0], [0], color="black", linestyle="--", label=f"mean {mean_rmse:.4f}")],
        loc="lower right",
        fontsize=8,
    )

    fig.suptitle(
        f"{title}\n"
        f"best  {best['label']}   RMSE={best['rmse_rad']:.4f}    "
        f"worst  {worst['label']}   RMSE={worst['rmse_rad']:.4f}",
        fontsize=12,
    )
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    from config.registry import get_task_config
    from experiment_play import PlaybackRunner

    checkpoint = resolve_checkpoint(args.checkpoint)
    cfg = get_task_config(args.task)
    output_dir = run_dir_for(checkpoint) / "playback_figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    start_idx = int(cfg.data.history_len) - 1
    dt = float(cfg.data.dt)
    run_frames = [
        pd.read_csv((REPO_ROOT / run_spec).resolve()).dropna().reset_index(drop=True)
        for run_spec in cfg.data.dataset.runs
    ]
    runner = PlaybackRunner(args.task, str(checkpoint))

    for run_index, (run_spec, df) in enumerate(zip(cfg.data.dataset.runs, run_frames)):
        print(f"[{run_index}] {run_spec}")
        sim_qpos, gt_qpos, _, _, _ = runner.run_index(
            run_index=run_index,
            df=df,
        )
        sim_qpos = np.asarray(sim_qpos)
        gt_qpos = np.asarray(gt_qpos)
        err = sim_qpos - gt_qpos

        csv_path = (REPO_ROOT / run_spec).resolve()
        qact = df["qact"].to_numpy(dtype=np.float64)[start_idx : start_idx + len(err)]

        figure_path = output_dir / f"{run_index:02d}_{csv_path.stem}.png"
        save_run_figure(
            figure_path=figure_path,
            title=f"run {run_index}: {run_spec}",
            dt=dt,
            sim_qpos=sim_qpos,
            gt_qpos=gt_qpos,
            qact=qact,
        )

        rows.append(
            {
                "run_index": run_index,
                "file": run_spec,
                "kind": motion_kind(csv_path.stem),
                "trajectory_len": len(err),
                "rmse_rad": rmse(err),
                "mae_rad": float(np.mean(np.abs(err))),
                "max_abs_rad": float(np.max(np.abs(err))),
                "max_abs_index": int(np.argmax(np.abs(err))),
                "figure": figure_path.name,
            }
        )

    metrics = pd.DataFrame(rows)
    metrics_path = output_dir / "metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    summary_path = output_dir / "summary.png"
    save_summary_figure(
        metrics=metrics,
        figure_path=summary_path,
        title=args.task,
    )
    print(
        "summary: "
        f"mean RMSE={metrics['rmse_rad'].mean():.4f} rad  "
        f"MAE={metrics['mae_rad'].mean():.4f} rad  "
        f"worst={int(metrics.loc[metrics['rmse_rad'].idxmax(), 'run_index'])} "
        f"({metrics['rmse_rad'].max():.4f} rad)"
    )
    print(f"saved figures, metrics, and {summary_path.name} to {output_dir}")


if __name__ == "__main__":
    main()
