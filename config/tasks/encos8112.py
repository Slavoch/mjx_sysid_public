from __future__ import annotations

import mujoco

from sysidmjx import models
from sysidmjx.distributions import Uniform
from sysidmjx.optimizers import run_single_motor_torque_opt

from config.base import (
    AssetConfig,
    ComponentConfig,
    DataConfig,
    DatasetSpec,
    ExperimentConfig,
    LoggingConfig,
    RunConfig,
    SimConfig,
    TrainConfig,
)


_HISTORY_LEN = 32
_PREDICTION_HORIZON = 64
_VAL_PREDICTION_HORIZON = 100
_DATASET_DIR = "assets/datasets/encos8112_4p25kg_servant_sg"
_DATA = DataConfig(
    dataset=DatasetSpec(
        model_xml="assets/robots/encos80/encos8112_test.xml",
        runs=(
            f"{_DATASET_DIR}/pd_chirp_qpos_amp0p2_0p1_1p5hz_kp17_kd1p5.csv",
            f"{_DATASET_DIR}/pd_chirp_qpos_amp0p4_0p1_1p5hz_kp17_kd1p5.csv",
            f"{_DATASET_DIR}/pd_chirp_qpos_amp0p5_0p1_1p5hz_kp30_kd1p5.csv",
            f"{_DATASET_DIR}/pd_chirp_qpos_amp0p6_0p1_1p5hz_kp17_kd1p5.csv",
            f"{_DATASET_DIR}/pd_chirp_qpos_amp0p7_0p1_1p5hz_kp17_kd1p5.csv",
            f"{_DATASET_DIR}/pd_linear_qpos_0_to_1p2_8s_kp6_kd0p8.csv",
            f"{_DATASET_DIR}/pd_linear_qpos_0_to_1p2_8s_kp12_kd0p8.csv",
            f"{_DATASET_DIR}/pd_linear_qpos_0_to_1p2_8s_kp20_kd0p8.csv",
            f"{_DATASET_DIR}/pd_linear_qpos_0_to_1p6_8s_kp20_kd0p8.csv",
            f"{_DATASET_DIR}/pd_sine_qpos_amp0p5_0p25hz_kp6_kd0p8.csv",
            f"{_DATASET_DIR}/pd_sine_qpos_amp1p0_0p5hz_kp10_kd1.csv",
            f"{_DATASET_DIR}/pd_sine_qpos_amp1p3_0p5hz_kp20_kd1.csv",
            f"{_DATASET_DIR}/pd_sine_qpos_amp1p6_0p5hz_kp20_kd1.csv",
            f"{_DATASET_DIR}/pd_steps_qpos_amp0p7_800ms_s1_kp20_kd2p0.csv",
            f"{_DATASET_DIR}/pd_steps_qpos_amp0p8_800ms_s1_kp30_kd0p8.csv",
            f"{_DATASET_DIR}/pd_steps_qpos_amp0p9_800ms_s1_kp20_kd2p0.csv",
            f"{_DATASET_DIR}/pd_steps_qpos_amp1p0_800ms_s1_kp10_kd0p8.csv",
            f"{_DATASET_DIR}/pd_steps_qpos_amp1p0_800ms_s1_kp20_kd0p8.csv",
            f"{_DATASET_DIR}/pd_steps_qpos_amp1p0_800ms_s7_kp20_kd0p8.csv",
            f"{_DATASET_DIR}/pd_steps_qpos_amp1p0_800ms_s8_kp20_kd0p8.csv",
            f"{_DATASET_DIR}/pd_steps_qpos_amp1p0_800ms_s9_kp25_kd0p8.csv",
            f"{_DATASET_DIR}/pd_hold_qpos0_kp6_kd0p8.csv",
        ),
        val_runs=(
            f"{_DATASET_DIR}/pd_chirp_qpos_amp0p2_0p1_1p5hz_kp17_kd1p5.csv",
            f"{_DATASET_DIR}/pd_chirp_qpos_amp0p4_0p1_1p5hz_kp17_kd1p5.csv",
            f"{_DATASET_DIR}/pd_chirp_qpos_amp0p5_0p1_1p5hz_kp30_kd1p5.csv",
            f"{_DATASET_DIR}/pd_chirp_qpos_amp0p6_0p1_1p5hz_kp17_kd1p5.csv",
            f"{_DATASET_DIR}/pd_chirp_qpos_amp0p7_0p1_1p5hz_kp17_kd1p5.csv",
            f"{_DATASET_DIR}/pd_linear_qpos_0_to_1p2_8s_kp6_kd0p8.csv",
            f"{_DATASET_DIR}/pd_linear_qpos_0_to_1p2_8s_kp12_kd0p8.csv",
            f"{_DATASET_DIR}/pd_linear_qpos_0_to_1p2_8s_kp20_kd0p8.csv",
            f"{_DATASET_DIR}/pd_linear_qpos_0_to_1p6_8s_kp20_kd0p8.csv",
            f"{_DATASET_DIR}/pd_sine_qpos_amp0p5_0p25hz_kp6_kd0p8.csv",
            f"{_DATASET_DIR}/pd_sine_qpos_amp1p0_0p5hz_kp10_kd1.csv",
            f"{_DATASET_DIR}/pd_sine_qpos_amp1p3_0p5hz_kp20_kd1.csv",
            f"{_DATASET_DIR}/pd_sine_qpos_amp1p6_0p5hz_kp20_kd1.csv",
            f"{_DATASET_DIR}/pd_steps_qpos_amp0p7_800ms_s1_kp20_kd2p0.csv",
            f"{_DATASET_DIR}/pd_steps_qpos_amp0p8_800ms_s1_kp30_kd0p8.csv",
            f"{_DATASET_DIR}/pd_steps_qpos_amp0p9_800ms_s1_kp20_kd2p0.csv",
            f"{_DATASET_DIR}/pd_steps_qpos_amp1p0_800ms_s1_kp10_kd0p8.csv",
            f"{_DATASET_DIR}/pd_steps_qpos_amp1p0_800ms_s1_kp20_kd0p8.csv",
            f"{_DATASET_DIR}/pd_steps_qpos_amp1p0_800ms_s7_kp20_kd0p8.csv",
            f"{_DATASET_DIR}/pd_steps_qpos_amp1p0_800ms_s8_kp20_kd0p8.csv",
            f"{_DATASET_DIR}/pd_steps_qpos_amp1p0_800ms_s9_kp25_kd0p8.csv",
            f"{_DATASET_DIR}/pd_hold_qpos0_kp6_kd0p8.csv",
        ),
        rollout_fields={
            "target_qpos": "qact",
            "target_qvel": "target_qvel",
            "kp": "kp",
            "kd": "kd",
        },
    ),
    dt=0.001,
    history_len=_HISTORY_LEN,
    prediction_horizon=_PREDICTION_HORIZON,
    val_prediction_horizon=_VAL_PREDICTION_HORIZON,
)


def pd_then_nn() -> RunConfig:
    return RunConfig(
        runner=run_single_motor_torque_opt,
        data=_DATA,
        sim=SimConfig(
            model_xml=_DATA.dataset.model_xml,
            integrator=mujoco.mjtIntegrator.mjINT_EULER,
            iterations=1,
        ),
        train=TrainConfig(
            seed=3,
            max_iter=1000,
            batch_size=1024,
            learning_rate=1e-2,
            learning_rate_final=1e-5,
            early_stop_patience=1000,
            early_stop_min_delta=0.0,
            loss_w_qpos=1.0,
            loss_w_qvel=1.0,
            loss_qpos_scale=1e-4,
            loss_qvel_scale=5e-2,
            loss_huber_delta=1.0,
        ),
        experiment=ExperimentConfig(
            name="Encos8112 dataset PD with learned residual torque",
            models=(
                ComponentConfig(
                    func=models.get_model_pd_residual,
                    params={
                        "hidden_dims": [32, 32],
                        "input_signals": ("tau_cmd", "qpos", "qvel"),
                        "history_len": _HISTORY_LEN,
                    },
                ),
            ),
            frictions=(),
            changers=(),
        ),
        assets=AssetConfig(),
        logging=LoggingConfig(
            runs_root="runs",
            log_every=50,
            checkpoint_every=400,
        ),
    )


def pd_then_isotropic_nn() -> RunConfig:
    cfg = pd_then_nn()
    return RunConfig(
        runner=cfg.runner,
        data=cfg.data,
        sim=cfg.sim,
        train=cfg.train,
        experiment=ExperimentConfig(
            name="Encos8112 dataset PD with odd-symmetric learned residual torque",
            models=(
                ComponentConfig(
                    func=models.get_model_isotropic_pd_residual,
                    params={
                        "hidden_dims": [32, 32],
                        "input_signals": ("tau_cmd", "qvel"),
                        "history_len": _HISTORY_LEN,
                    },
                ),
            ),
            frictions=cfg.experiment.frictions,
            changers=cfg.experiment.changers,
        ),
        assets=cfg.assets,
        logging=cfg.logging,
    )


def pd_then_isotropic_nn_frictionloss() -> RunConfig:
    cfg = pd_then_isotropic_nn()
    return RunConfig(
        runner=cfg.runner,
        data=cfg.data,
        sim=cfg.sim,
        train=cfg.train,
        experiment=ExperimentConfig(
            name="Encos8112 dataset PD with odd-symmetric residual torque and learned frictionloss",
            models=cfg.experiment.models,
            frictions=cfg.experiment.frictions,
            changers=(
                *cfg.experiment.changers,
                ComponentConfig(
                    func=models.make_changer_frictionloss,
                    params={"frictionloss_init": Uniform(min=0.9, max=1.1)},
                ),
            ),
        ),
        assets=cfg.assets,
        logging=cfg.logging,
    )


def pd_then_changers() -> RunConfig:
    cfg = pd_then_nn()
    return RunConfig(
        runner=cfg.runner,
        data=cfg.data,
        sim=cfg.sim,
        train=cfg.train,
        experiment=ExperimentConfig(
            name="Encos8112 dataset PD with learned plant changers",
            models=(
                ComponentConfig(
                    func=models.get_model_pd_control,
                ),
            ),
            frictions=(),
            changers=(
                ComponentConfig(
                    func=models.make_changer_frictionloss,
                    params={"frictionloss_init": Uniform(min=0.0, max=0.1)},
                ),
                ComponentConfig(
                    func=models.make_changer_damping,
                    params={"damping_init": Uniform(min=0.0, max=0.1)},
                ),
                ComponentConfig(
                    func=models.make_changer_armature,
                    params={"armature_init": Uniform(min=0.0, max=0.01)},
                ),
            ),
        ),
        assets=cfg.assets,
        logging=cfg.logging,
    )


def pd_then_changers_friction_solver() -> RunConfig:
    cfg = pd_then_changers()
    return RunConfig(
        runner=cfg.runner,
        data=cfg.data,
        sim=cfg.sim,
        train=cfg.train,
        experiment=ExperimentConfig(
            name="Encos8112 dataset PD with learned plant and friction solver changers",
            models=cfg.experiment.models,
            frictions=cfg.experiment.frictions,
            changers=(
                *cfg.experiment.changers,
                ComponentConfig(
                    func=models.make_changer_solreffriction,
                    params={"solreffriction_init": (0.02, 1.0)},
                ),
                ComponentConfig(
                    func=models.make_changer_solimpfriction,
                    params={"solimpfriction_init": (0.9, 0.95, 0.001, 0.5, 2.0)},
                ),
            ),
        ),
        assets=cfg.assets,
        logging=cfg.logging,
    )


def pd_then_karnopp_stribeck() -> RunConfig:
    cfg = pd_then_nn()
    return RunConfig(
        runner=cfg.runner,
        data=cfg.data,
        sim=cfg.sim,
        train=cfg.train,
        experiment=ExperimentConfig(
            name="Encos8112 dataset PD with learned Karnopp-Stribeck friction",
            models=(
                ComponentConfig(
                    func=models.get_model_pd_control,
                ),
            ),
            frictions=(
                ComponentConfig(
                    func=models.make_friction_karnopp_stribeck,
                    params={
                        "coulomb_init": Uniform(min=0.0, max=0.1),
                        "static_init": Uniform(min=0.0, max=0.1),
                        "stribeck_vel_init": Uniform(min=0.001, max=0.1),
                    },
                ),
            ),
            changers=(
                ComponentConfig(
                    func=models.make_changer_damping,
                    params={"damping_init": Uniform(min=0.0, max=0.1)},
                ),
                ComponentConfig(
                    func=models.make_changer_armature,
                    params={"armature_init": Uniform(min=0.0, max=0.01)},
                ),
            ),
        ),
        assets=cfg.assets,
        logging=cfg.logging,
    )
