from __future__ import annotations

import mujoco
from sysidmjx import models as controllers
from sysidmjx.optimizers import run_single_motor_torque_opt

from config.base import (
    AssetConfig,
    ComponentConfig,
    DataConfig,
    ExperimentConfig,
    LoggingConfig,
    SimConfig,
    TrainConfig,
    RunConfig,
)


def baseline() -> RunConfig:
    return RunConfig(
        runner=run_single_motor_torque_opt,
        data=DataConfig(
            dataset_path="",
            dt=0.002,
            motor_id=0,
            num_lags=1,
        ),
        sim=SimConfig(
            model_xml="",
            integrator=mujoco.mjtIntegrator.mjINT_EULER,
            iterations=1,
        ),
        train=TrainConfig(),
        experiment=ExperimentConfig(
            name="Minipi baseline",
            controllers=(ComponentConfig(func=controllers.get_controller_optimal_torque),),
        ),
        assets=AssetConfig(),
        logging=LoggingConfig(),
    )
