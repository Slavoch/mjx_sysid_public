# MuJoCo plant model

[← Documentation overview](../README.md#documentation)

The XML describes the physical plant around the actuator: the body, hinge, load, inertial properties, and torque actuator. System identification can compensate for uncertain actuator dynamics, but it cannot recover a badly measured load or an incorrect kinematic model.

The current pipeline supports a single actuated hinge (one degree of freedom).

## Model requirements

- Use one hinge joint and one actuator.
- Match the bench setup's mass, center of mass, geometry, and inertia as closely as practical.
- Set the simulation timestep through `DataConfig.dt`; the optimizer applies it to the model.
- Treat `SimConfig.integrator` and `SimConfig.iterations` as authoritative during training. They override the corresponding XML options; the bundled tasks use Euler integration with one solver iteration.
- Start parameters you intend to identify near zero. A large fixed value in the XML can interfere with or duplicate a learned contribution.

The bundled example is [`assets/robots/encos80/encos8112_test.xml`](../assets/robots/encos80/encos8112_test.xml).

## Minimal structure

```xml
<mujoco model="single_motor">
  <worldbody>
    <body name="motor" pos="0 0 0.5">
      <body name="load">
        <joint
          name="hinge"
          type="hinge"
          axis="0 -1 0"
          frictionloss="0.00001"
        />
        <geom type="box" size="..." mass="..." />
        <inertial pos="..." mass="..." diaginertia="..." />
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor
      name="hinge"
      joint="hinge"
      gear="1"
      ctrllimited="true"
      ctrlrange="-90 90"
    />
  </actuator>
</mujoco>
```

Replace the placeholder geometry and inertial values with measurements from your setup. The complete example also includes the visual and scene configuration used by the playback tools.

## Parameters learned outside the XML

Plant changers overwrite selected MJX fields during optimization. Available changers include:

- joint `armature`, `damping`, and `frictionloss`;
- actuator gain and filter time constant;
- friction solver `solref` and `solimp` values, stored in MJX as `dof_solref` and `dof_solimp`.

Define their initial values in the task's `ComponentConfig`; see [Motor models](motor_models.md). Keep the corresponding XML values neutral so that there is one clear source for each learned effect.

## Before training

1. Open the model with MuJoCo and confirm that it compiles.
2. Check the joint axis and sign convention against the logged `qpos` and `qvel`.
3. Verify that the resting pose and gravity response match the physical rig.
4. Confirm that actuator limits cover the commands in the dataset.

Once the plant is correct, connect it to trajectory files through `DatasetSpec`; see [Datasets](datasets.md).

The side-by-side validation viewer generates `<model_stem>_twin.xml` next to the source model at runtime; this twin file is derived output rather than a separate plant definition.