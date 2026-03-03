from __future__ import annotations

from dataclasses import replace
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
from sysidmjx.distributions import Normal, Uniform


def torque() -> RunConfig:
    return RunConfig(
        runner=run_single_motor_torque_opt,
        data=DataConfig(
            dataset_path="assets/datasets/roki2/generated/test.csv",
            dt=0.002,
            motor_id=0,
            num_lags=3,
        ),
        sim=SimConfig(
            model_xml="assets/robots/roki2/motor_roki.xml",
            integrator=mujoco.mjtIntegrator.mjINT_EULER,
            iterations=1,
        ),
        train=TrainConfig(
            seed=3,
            max_iter=5000,
            batch_size=20000,
            learning_rate=1e-2,
            early_stop_patience=200,
            early_stop_min_delta=0.0,
            loss_w_qpos=1.0,
            loss_w_qvel=0.0,
        ),
        experiment=ExperimentConfig(
            name="Optimal torque",
            controllers=(
                ComponentConfig(
                    func=controllers.get_controller_optimal_torque,
                    params={"initialization": "zeros_by_dataset"},
                ),
            ),
            frictions=(),
            changers=(),
        ),
        assets=AssetConfig(),
        logging=LoggingConfig(
            runs_root="runs",
            log_every=1,
            checkpoint_every=400,
        ),
    )


def _with_experiment(base: RunConfig, name: str, controllers_cfg, frictions_cfg=(), changers_cfg=()) -> RunConfig:
    return replace(
        base,
        experiment=ExperimentConfig(
            name=name,
            controllers=controllers_cfg,
            frictions=frictions_cfg,
            changers=changers_cfg,
        ),
    )

def karnopp_smooth() -> RunConfig:
    baseline = torque()
    return _with_experiment(
        baseline,
        name="Karnopp smooth model",
        controllers_cfg=(
            ComponentConfig(
                func=controllers.get_controller_pd_fixed,
                params={"kp": 20.0, "kv": 1.0},
            ),
        ),
        frictions_cfg=(
            ComponentConfig(
                func=controllers.make_friction_karnopp,
                params={"friction_init": Uniform(min=0.0, max=0.01)},
            ),
        ),
        changers_cfg=(
            ComponentConfig(
                func=controllers.make_changer_armature,
                params={"armature_init": Uniform(min=0.0, max=0.01)},
            ),
        ),
    )


def nn32_pd_friction() -> RunConfig:
    baseline = torque()
    return _with_experiment(
        baseline,
        name="NN 32 model (PD + friction NN)",
        controllers_cfg=(
            ComponentConfig(
                func=controllers.get_controller_pd_fixed,
                params={"kp": 20.0, "kv": 1.0},
            ),
        ),
        frictions_cfg=(
            ComponentConfig(
                func=controllers.make_friction_nn,
                params={"dims": [4, 32, 32, 1]},
            ),
        ),
        changers_cfg=(
            ComponentConfig(
                func=controllers.make_changer_armature,
                params={"armature_init": Uniform(min=0.0, max=0.01)},
            ),
        ),
    )


def nn32_ctrl_and_friction() -> RunConfig:
    baseline = torque()
    return _with_experiment(
        baseline,
        name="NN 32 model (controller NN + friction NN)",
        controllers_cfg=(
            ComponentConfig(
                func=controllers.get_controller_nn,
                params={"dims": [3, 32, 32, 1]},
            ),
        ),
        frictions_cfg=(
            ComponentConfig(
                func=controllers.make_friction_nn,
                params={"dims": [4, 32, 32, 1]},
            ),
        ),
        changers_cfg=(
            ComponentConfig(
                func=controllers.make_changer_armature,
                params={"armature_init": Uniform(min=0.0, max=0.01)},
            ),
        ),
    )


def pd_learning() -> RunConfig:
    baseline = torque()
    return _with_experiment(
        baseline,
        name="PD learning",
        controllers_cfg=(
            ComponentConfig(
                func=controllers.get_controller_pd_variable,
                params={
                    "kp_init": Normal(mean=1, std=0.25),
                    "kv_init": Normal(mean=1, std=0.25),
                },
            ),
        ),
        frictions_cfg=(),
        changers_cfg=(
            ComponentConfig(
                func=controllers.make_changer_armature,
                params={"armature_init": Uniform(min=0.0, max=0.01)},
            ),
            # ComponentConfig(func=controllers.make_changer_frictionloss),
        ),
    )


def nn32_with_changers() -> RunConfig:
    baseline = torque()
    return _with_experiment(
        baseline,
        name="NN 32 model",
        controllers_cfg=(
            ComponentConfig(
                func=controllers.get_controller_nn,
                params={"dims": [3, 32, 32, 1]},
            ),
        ),
        frictions_cfg=(),
        changers_cfg=(
            ComponentConfig(
                func=controllers.make_changer_armature,
                params={"armature_init": Uniform(min=0.0, max=0.01)},
            ),
        ),
    )