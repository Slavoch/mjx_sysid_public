from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"time_s", "qpos", "qvel"}
DEFAULT_MODEL_XML = Path("assets/robots/encos80/encos8112_test.xml")
PD_COLUMNS = {"qpos", "qvel", "qact", "kp", "kd"}
QVEL_DES_COLUMNS = ("target_qvel", "qvel_des")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add inverse-dynamics required torque to dataset CSV files."
    )
    parser.add_argument("input", type=Path, help="Folder containing CSV files.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output folder. Defaults to '<input>_gt' next to the input folder.",
    )
    parser.add_argument(
        "--model-xml",
        type=Path,
        default=DEFAULT_MODEL_XML,
        help=f"MuJoCo XML used for inverse dynamics. Default: {DEFAULT_MODEL_XML}",
    )
    return parser.parse_args()


def default_output_dir(input_dir: Path) -> Path:
    return input_dir.with_name(f"{input_dir.name}_gt")


def validate_frame(df: pd.DataFrame, csv_path: Path) -> None:
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")


def required_torque(df: pd.DataFrame, model: mujoco.MjModel) -> np.ndarray:
    data = mujoco.MjData(model)
    time_s = df["time_s"].to_numpy(dtype=np.float64)
    qpos = df["qpos"].to_numpy(dtype=np.float64)
    qvel = df["qvel"].to_numpy(dtype=np.float64)
    qacc = np.gradient(qvel, time_s)

    torque = np.empty_like(qpos)
    for i, (q, v, a) in enumerate(zip(qpos, qvel, qacc)):
        data.qpos[0] = q
        data.qvel[0] = v
        data.qacc[0] = a
        mujoco.mj_inverse(model, data)
        torque[i] = data.qfrc_inverse[0]
    return torque


def desired_velocity_column(df: pd.DataFrame) -> str | None:
    for column in QVEL_DES_COLUMNS:
        if column in df.columns:
            return column
    return None


def can_compute_sent_torque(df: pd.DataFrame) -> bool:
    return PD_COLUMNS.issubset(df.columns) and desired_velocity_column(df) is not None


def sent_torque(df: pd.DataFrame) -> np.ndarray:
    qvel_des = desired_velocity_column(df)
    if qvel_des is None:
        raise ValueError(f"Missing desired velocity column, expected one of {QVEL_DES_COLUMNS}")
    return (
        df["kp"].to_numpy(dtype=np.float64)
        * (df["qact"].to_numpy(dtype=np.float64) - df["qpos"].to_numpy(dtype=np.float64))
        + df["kd"].to_numpy(dtype=np.float64)
        * (df[qvel_des].to_numpy(dtype=np.float64) - df["qvel"].to_numpy(dtype=np.float64))
    )


def save_figure(df: pd.DataFrame, figure_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    time_s = df["time_s"].to_numpy(dtype=np.float64)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(title)

    axes[0].plot(time_s, df["qpos"], label="qpos")
    if "qact" in df.columns:
        axes[0].plot(time_s, df["qact"], label="qact")
    axes[0].set_ylabel("position [rad]")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(time_s, df["qvel"], label="qvel")
    axes[1].set_ylabel("qvel [rad/s]")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(time_s, df["torque"], label="required torque", color="tab:red")
    if "sent_torque" in df.columns:
        axes[2].plot(time_s, df["sent_torque"], label="sent torque (PD)", color="tab:blue", alpha=0.8)
    axes[2].set_xlabel("time [s]")
    axes[2].set_ylabel("torque")
    axes[2].legend()
    axes[2].grid(True)

    fig.tight_layout()
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_dir = args.input.resolve()
    output_dir = default_output_dir(input_dir) if args.output is None else args.output.resolve()
    figures_dir = output_dir / "figures"
    model_xml = args.model_xml.resolve()

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")
    if not model_xml.is_file():
        raise FileNotFoundError(f"Model XML does not exist: {model_xml}")

    csv_files = sorted(path for path in input_dir.glob("*.csv") if path.is_file())
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(model_xml))

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        validate_frame(df, csv_path)
        df["torque"] = required_torque(df, model)
        if can_compute_sent_torque(df):
            df["sent_torque"] = sent_torque(df)
        output_csv = output_dir / csv_path.name
        figure_path = figures_dir / f"{csv_path.stem}.png"
        df.to_csv(output_csv, index=False)
        save_figure(df, figure_path, csv_path.name)
        print(f"{csv_path} -> {output_csv}")


if __name__ == "__main__":
    main()
