# Motor models

[← Documentation overview](../README.md#documentation)

Motor behavior is assembled from three ordered component types in [`sysidmjx/models.py`](../sysidmjx/models.py):

1. **Action models** transform logged commands and history into torque.
2. **Friction models** compute torque opposing motion and subtract it from the action.
3. **Plant changers** overwrite selected fields of the MJX model.

Start with the smallest model that can explain the data. For many motors, learned `frictionloss`, `damping`, and `armature` are a useful baseline. Add a richer friction law or neural residual only when full-trajectory validation shows a consistent mismatch.

## Available components

### Action models

| Factory | Purpose |
| --- | --- |
| `get_model_action_passthrough` | Treat `qact` as torque without learning a mapping |
| `get_model_pd_fixed` | Convert position targets to torque with fixed `kp` and `kv` |
| `get_model_pd_variable` | Learn `kp` and `kv` for a PD mapping |
| `get_model_pd_control` | Reconstruct PD torque from logged targets and gains |
| `get_model_torque_constant_from_qvel` | Learn a velocity-dependent command-to-torque scale |
| `get_model_nn` | Predict torque with an MLP over signal history |
| `get_model_nn_torque_residual` | Add a learned residual to command torque |
| `get_model_nn_qvel_qact_residual` | Learn a residual from velocity and action history |
| `get_model_nn_qvel_qact_residual_normalized` | Normalized velocity/action residual model |
| `get_model_pd_residual` | Add an MLP residual to reconstructed PD torque |
| `get_model_isotropic_pd_residual` | Add an odd-symmetric residual, `f(x) - f(-x)` |
| `get_model_optimal_torque` | Use an explicitly supplied torque sequence |

### Friction models

| Factory | Purpose |
| --- | --- |
| `make_friction_karnopp` | Karnopp friction around zero velocity |
| `make_friction_smooth_karnopp` | Smooth Karnopp approximation |
| `make_friction_karnopp_stribeck` | Karnopp model with a Stribeck curve |
| `make_friction_nn` | Learned friction function |

### Plant changers

| Factory | MJX field or behavior |
| --- | --- |
| `make_changer_armature` | `dof_armature` |
| `make_changer_damping` | `dof_damping` |
| `make_changer_frictionloss` | `dof_frictionloss` |
| `make_changer_solreffriction` | friction `solref` |
| `make_changer_solimpfriction` | friction `solimp` |
| `make_changer_actuator_filter_timeconst` | actuator filter time constant |
| `make_changer_actuator_gain` | actuator gain |

Each factory returns an initialized component and its parameter tree. Initialization values can be fixed scalars or vectors, or sampled from `Uniform` and `Normal` distributions.

## Common configurations

### Logged PD controller with learned plant parameters

Use `get_model_pd_control` when each CSV contains the target position, target velocity, `kp`, and `kd`. Combine it with plant changers:

```python
experiment=ExperimentConfig(
    name="PD with learned plant parameters",
    models=(
        ComponentConfig(func=models.get_model_pd_control),
    ),
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
)
```

The dataset mapping must expose the fields expected by the controller; see [Datasets](datasets.md).

### Unknown fixed PD gains

When `qact` is a position target but the command-to-torque mapping is unknown, learn the gains:

```python
ComponentConfig(
    func=models.get_model_pd_variable,
    params={
        "kp_init": Uniform(min=0.0, max=30.0),
        "kv_init": Uniform(min=0.0, max=5.0),
    },
)
```

This implements:

```text
torque = kp * (qact - qpos) - kv * qvel
```

Use `get_model_pd_fixed(kp=..., kv=...)` when both gains are known.

### Residual dynamics

`get_model_pd_residual` reconstructs the PD command and adds an MLP correction. `get_model_isotropic_pd_residual` constrains that correction to be odd-symmetric, which can be useful when reversing position, velocity, and torque should reverse the residual.

Set the component's `history_len` to the same value as `DataConfig.history_len`.

## Bundled tasks

A task is a `RunConfig` factory in `config/tasks/` registered under a CLI key in [`config/registry.py`](../config/registry.py).

| Task key | Identifies |
| --- | --- |
| `encos8112.pd_then_changers` | `frictionloss`, `damping`, and `armature` |
| `encos8112.pd_then_changers_friction_solver` | Baseline parameters plus friction `solref` and `solimp` |
| `encos8112.pd_then_karnopp_stribeck` | Karnopp–Stribeck friction, damping, and armature |
| `encos8112.pd_then_nn` | PD torque plus an MLP residual |
| `encos8112.pd_then_isotropic_nn` | PD torque plus an odd-symmetric residual |
| `encos8112.pd_then_isotropic_nn_frictionloss` | Odd-symmetric residual plus `frictionloss` |

List these keys directly from the registry:

```bash
python scripts/optimize.py --list-tasks
```

## Add a task

1. Define a function returning `RunConfig` in `config/tasks/`.
2. Select the dataset, simulation settings, training settings, and model components.
3. Add the function to `TASK_REGISTRY` in `config/registry.py`.
4. Confirm that the key appears under `--list-tasks`.
5. Run it with `python scripts/optimize.py --task=<your.key>`.

See [Learning](learning.md) for rollout and optimizer settings.
