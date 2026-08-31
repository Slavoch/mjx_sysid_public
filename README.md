# MJX System Identification

Calibrate a single-motor MuJoCo model from motion logs—without a torque sensor.

MJX System Identification differentiates through simulated trajectories to find the actuator parameters that make simulation follow real hardware. It can fit MuJoCo parameters such as friction loss, damping, and armature, or learn richer actuator behavior with friction laws and neural-network residuals.

## Before and after identification

| Before | After |
| --- | --- |
| ![Simulation before system identification](docs/ezgif-1817e477473ca576.gif) | ![Simulation after system identification](docs/ezgif-7c91bd5f79225753.gif) |

The green arm follows the recorded hardware trajectory; the other arm is the simulated model. Watch the [full demonstration video](docs/IROS25_1428_VI_fi.mp4).

## What you provide

- A single-DoF [MuJoCo XML](docs/xml.md) model of the motor and its load.
- One or more CSV [trajectories](docs/datasets.md) with `time_s`, `qpos`, `qvel`, and `qact`.
- Any signals needed to reconstruct the control law, such as PD targets and gains.

## What you can identify

- Built-in MuJoCo parameters: `frictionloss`, `damping`, `armature`, actuator gain, filter constants, and friction-solver settings.
- Parametric friction models, including Karnopp and Stribeck variants.
- PD mappings when the relationship between commands and torque is unknown.
- Learned torque or PD residuals for dynamics that a compact parametric model cannot capture.

See [Motor models](docs/motor_models.md) for the available components and bundled experiments.

## Quick start

Python 3.10 or newer is required; the example below uses Python 3.11. Run the commands from the repository root.

```bash
python3.11 -m venv venv
source venv/bin/activate

pip install -e .
```

This installs the default CPU build of JAX. For NVIDIA GPU acceleration with CUDA 12, additionally run:

```bash
pip install -U "jax[cuda12]"
```

The bundled tasks use the filtered Encos8112 dataset. Generate it from the tracked raw trajectories before the first run:

```bash
python scripts/filter_motion_sg.py assets/datasets/encos8112_4p25kg_servant
```

Then list the configured experiments and run the baseline parameter identification:

```bash
python scripts/optimize.py --list-tasks
python scripts/optimize.py --task=encos8112.pd_then_changers
```

Each run is written to:

```text
runs/<task>/<timestamp>/
├── config.json
├── result.json
├── best_params.pkl
├── loss_history.npy
├── grad_norm_history.npy
├── events.out.tfevents.*
└── checkpoints/
    └── epoch_<step>.pkl
```

Monitor training with TensorBoard:

```bash
tensorboard --logdir runs
```

## Identify your own motor

1. **Collect representative trajectories.** Cover the positions, velocities, and commands the motor will encounter. See [Datasets](docs/datasets.md).
2. **Model the physical plant.** Describe the motor, joint, and measured load in a single-DoF [MuJoCo XML](docs/xml.md).
3. **Choose the actuator model.** Start with friction loss, damping, and armature before moving to richer [motor models](docs/motor_models.md).
4. **Create a task.** Combine the XML, trajectories, model components, and optimization settings in a `RunConfig`, then register it in `config/registry.py`.
5. **Train and validate.** Use independent trajectories where available, then inspect full-sequence replay and aggregate plots. See [Learning](docs/learning.md) and [Validation](docs/validation.md).

If your measurements are noisy, the repository includes a Savitzky–Golay preprocessing script:

```bash
python scripts/filter_motion_sg.py path/to/dataset
```

## How it works

Training samples short windows from the recorded trajectories, initializes the simulation from the measured state, and rolls the MuJoCo plant forward under the recorded commands. JAX gradients update the selected actuator and plant parameters to reduce position and velocity error over the prediction horizon.

For learned parameters $z$, the rollout objective is:

$$
\mathcal{L}(z)
=
\frac{1}{MP}
\sum_{j=0}^{M-1}
\sum_{i=1}^{P}
\left\|
W\left(\hat{s}_{i,j}-s_{i,j}\right)
\right\|_2^2,
$$

subject to:

$$
\hat{s}_{0,j}=s_{0,j},
\qquad
\hat{s}_{i+1,j}
=
\Phi_z\left(\hat{s}_{i,j},a_{i,j};\Delta t\right).
$$

Here, $M$ is the number of sampled trajectory windows, $P$ is the prediction horizon, $s=(q,\dot q)$ is the measured state, $\hat{s}$ is the simulated state, $a$ is the recorded command, and $\Phi_z$ is one MJX simulation step. The diagonal matrix $W$ sets the relative position and velocity scales and weights:

$$
W =
\begin{bmatrix}
w_q & 0 \\
0 & w_{\dot q}
\end{bmatrix}.
$$

The implementation can replace the squared pointwise penalty with a configurable pseudo-Huber penalty for greater robustness to outliers.

During training, validation uses longer windows from `DatasetSpec.val_runs`; the best windowed validation result becomes `best_params.pkl`. Post-training playback replays complete command sequences to expose drift that windowed loss can hide.

For implementation details, see [Architecture](docs/architecture.md).

## Documentation

- [Dataset format and collection](docs/datasets.md)
- [MuJoCo plant model](docs/xml.md)
- [Actuator and friction models](docs/motor_models.md)
- [Optimization settings](docs/learning.md)
- [Replay and validation](docs/validation.md)
- [Code architecture](docs/architecture.md)

## Research

This repository accompanies work presented at IROS 2025 and in IEEE Access (2026). If you use it in your research, please cite:

```bibtex
@inproceedings{kovalev2025achieving,
  title={Achieving Precise and Reliable Locomotion with Differentiable Simulation-Based System Identification},
  author={Kovalev, Vyacheslav and Chaikovskaia, Ekaterina and Davydenko, Egor and Gorbachev, Roman},
  booktitle={2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={4767--4772},
  year={2025},
  organization={IEEE}
}

@article{kovalev2026trajectory,
  title={Trajectory-based actuator identification via differentiable simulation},
  author={Kovalev, Vyacheslav and Chaikovskaia, Ekaterina and Davydenko, Egor and Gorbachev, Roman},
  journal={IEEE Access},
  year={2026},
  publisher={IEEE}
}
```

## License

Licensed under the [Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for attribution.
