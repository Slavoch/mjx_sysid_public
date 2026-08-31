from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


REQUIRED_COLUMNS = {"time_s", "qpos", "qvel", "qact"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply Savitzky-Golay filtering to motion CSV files."
    )
    parser.add_argument("input", type=Path, help="Input CSV file or folder of CSV files.")
    parser.add_argument("--window", type=int, default=111, help="Odd SG window length.")
    parser.add_argument("--polyorder", type=int, default=3, help="SG polynomial order.")
    return parser.parse_args()


def input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".csv":
            raise ValueError(f"Input file must be a CSV: {input_path}")
        return [input_path]
    if input_path.is_dir():
        return sorted(path for path in input_path.glob("*.csv") if path.is_file())
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def output_root_for(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_sg")


def validate_frame(df: pd.DataFrame, csv_path: Path) -> None:
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")


def validate_sg_params(n: int, window: int, polyorder: int) -> None:
    if window % 2 == 0:
        raise ValueError("--window must be odd")
    if window <= polyorder:
        raise ValueError("--window must be greater than --polyorder")
    if window > n:
        raise ValueError(f"--window={window} is larger than data length {n}")


def filter_csv(csv_path: Path, output_csv: Path, figure_path: Path, window: int, polyorder: int) -> None:
    df = pd.read_csv(csv_path)
    validate_frame(df, csv_path)
    validate_sg_params(len(df), window, polyorder)

    time_s = df["time_s"].to_numpy(dtype=np.float64)
    qpos_raw = df["qpos"].to_numpy(dtype=np.float64)
    qvel_raw = df["qvel"].to_numpy(dtype=np.float64)
    qact = df["qact"].to_numpy(dtype=np.float64)
    dt = float(np.median(np.diff(time_s)))
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"{csv_path} has invalid time_s spacing")

    qpos_sg = savgol_filter(qpos_raw, window_length=window, polyorder=polyorder, mode="interp")
    qvel_sg = savgol_filter(
        qpos_raw,
        window_length=window,
        polyorder=polyorder,
        deriv=1,
        delta=dt,
        mode="interp",
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df = df.copy()
    output_df["qpos"] = qpos_sg
    output_df["qvel"] = qvel_sg
    output_df.to_csv(output_csv, index=False)

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    save_summary_figure(
        figure_path=figure_path,
        title=csv_path.name,
        time_s=time_s,
        qpos_raw=qpos_raw,
        qpos_sg=qpos_sg,
        qvel_raw=qvel_raw,
        qvel_sg=qvel_sg,
        qact=qact,
    )


def save_summary_figure(
    *,
    figure_path: Path,
    title: str,
    time_s: np.ndarray,
    qpos_raw: np.ndarray,
    qpos_sg: np.ndarray,
    qvel_raw: np.ndarray,
    qvel_sg: np.ndarray,
    qact: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    raw_integrated_error = integration_error(qpos_raw, qvel_raw, time_s)
    sg_integrated_error = integration_error(qpos_raw, qvel_sg, time_s)

    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    fig.suptitle(title)

    axes[0].plot(time_s, qpos_raw, label="raw qpos", linewidth=1.0)
    axes[0].plot(time_s, qpos_sg, label="sg qpos", linewidth=1.0)
    axes[0].set_ylabel("qpos [rad]")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(time_s, qvel_raw, label="raw qvel", linewidth=1.0)
    axes[1].plot(time_s, qvel_sg, label="sg qvel", linewidth=1.0)
    axes[1].set_ylabel("qvel [rad/s]")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(time_s, qact, label="qact", linewidth=1.0)
    axes[2].set_ylabel("qact")
    axes[2].legend()
    axes[2].grid(True)

    axes[3].plot(time_s, raw_integrated_error, label="integrated raw qvel - raw qpos", linewidth=1.0)
    axes[3].plot(time_s, sg_integrated_error, label="integrated sg qvel - raw qpos", linewidth=1.0)
    axes[3].axhline(0.0, color="black", linewidth=0.8)
    axes[3].set_xlabel("time [s]")
    axes[3].set_ylabel("qvel int. error [rad]")
    axes[3].legend()
    axes[3].grid(True)

    fig.tight_layout()
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)


def integration_error(qpos: np.ndarray, qvel: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    dt = np.diff(time_s, prepend=time_s[0])
    qpos_from_qvel = qpos[0] + np.cumsum(qvel * dt)
    return qpos_from_qvel - qpos


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    files = input_files(input_path)
    output_root = output_root_for(input_path)
    figures_root = output_root / "figures"

    if not files:
        print(f"No CSV files found in {input_path}")
        return

    for csv_path in files:
        output_csv = output_root / csv_path.name
        figure_path = figures_root / f"{csv_path.stem}.png"
        filter_csv(csv_path, output_csv, figure_path, args.window, args.polyorder)
        print(f"{csv_path} -> {output_csv}")


if __name__ == "__main__":
    main()
