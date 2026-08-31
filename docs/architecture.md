# Architecture

[← Documentation overview](../README.md#documentation)

The project separates experiment configuration from the reusable MJX identification pipeline. A task selects data, simulation settings, model components, and training settings; the optimizer then builds and differentiates the complete rollout.

## Repository layout

```text
config/
├── base.py          Configuration dataclasses
├── registry.py      CLI task keys
└── tasks/           Experiment factories

sysidmjx/
├── assets/          Single-motor data loading
├── optimizers/      Training loop and component construction
├── core.py          Differentiable rollout and loss functions
├── models.py        Action, friction, and plant components
├── distributions.py Parameter initialization specifications
└── utils/xml_clone.py Twin-model generation for playback

scripts/
├── optimize.py          Training entry point
├── dataset_play.py      Recorded-state viewer
├── experiment_play.py   Dataset-versus-simulation viewer
├── plot_task_runs.py    Full-trajectory metrics and plots
├── filter_motion_sg.py  Dataset preprocessing
└── mark_required_torque.py Inverse-dynamics dataset diagnostics

assets/
├── robots/          MuJoCo XML models
└── datasets/        Recorded trajectories

runs/                Generated training outputs
```

## Configuration model

`RunConfig` is the complete experiment definition:

| Section | Responsibility |
| --- | --- |
| `data: DataConfig` | Dataset paths, timestep, history, and rollout horizons |
| `sim: SimConfig` | MuJoCo integrator and solver iterations |
| `train: TrainConfig` | Optimizer, loss, and early-stopping settings |
| `experiment: ExperimentConfig` | Action models, friction models, and plant changers |
| `logging: LoggingConfig` | Output root and logging/checkpoint intervals |
| `assets: AssetConfig` | Asset directory conventions; currently serialized but not consumed by the runner |
| `runner` | Function that executes the configured optimization |

`DatasetSpec` links the plant XML to training and validation CSVs. Its `rollout_fields` mapping exposes additional CSV columns to action models.

Task factories live in `config/tasks/`. `TASK_REGISTRY` gives each factory a stable CLI name.

## Training flow

Running:

```bash
python scripts/optimize.py --task=<task.key>
```

executes the following pipeline:

1. Resolve the task factory from `config/registry.py`.
2. Create `runs/<task>/<timestamp>/` and save the resolved configuration.
3. Load each CSV, drop missing rows, validate required fields, and construct overlapping history/target windows.
4. Initialize the configured action, friction, and changer parameter trees.
5. Load the MuJoCo XML, apply simulation settings, and transfer the model to MJX.
6. Compose an action function and model-changing function from the configured components.
7. Roll each sampled window forward and compute scaled position and velocity losses.
8. Differentiate the loss with JAX and update parameters with Adam.
9. Evaluate all train and validation windows at logging intervals.
10. Save the best validation parameters and periodic checkpoints.

## Component composition

`compose_action_model` applies components in a fixed order:

```text
logged state and commands
        │
        ▼
action model chain
        │
        ▼
candidate torque ── friction models ──► applied torque

base MJX model ── plant changers ──► rollout model
```

Action models transform `ModelState`, which contains position, velocity, and action history plus rollout-time control fields. Friction components are summed and subtracted from the resulting torque. Changers update the MJX model before rollout.

All learned values are stored in a nested parameter tree:

```text
params/
├── models/
├── frictions/
└── changers/
```

## Outputs

Each optimization run contains:

- `config.json` — resolved, serializable task configuration;
- `result.json` — final status and summary metrics;
- `best_params.pkl` — parameters with the best windowed validation loss;
- `checkpoints/epoch_<step>.pkl` — periodic snapshots;
- `loss_history.npy` and `grad_norm_history.npy`;
- TensorBoard event files.

Playback plots and metrics are added by `scripts/plot_task_runs.py`; see [Validation](validation.md).

## Extension points

- Add a dataset by creating CSVs and referencing them in `DatasetSpec`.
- Add an experiment by defining and registering a new `RunConfig` factory.
- Add actuator behavior by implementing a factory in `sysidmjx/models.py`.
- Add a new optimization procedure by assigning another callable to `RunConfig.runner`.

Keep reusable mechanics under `sysidmjx/` and task-specific choices under `config/tasks/`.
