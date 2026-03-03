import jax
import jax.numpy as jnp
from jax import Array
from typing import List, Dict, Any, Tuple, Callable
from typing import Optional
from mujoco import mjx
from sysidmjx.distributions import Distribution

# Each factory has signature: def factory(seed: Array) -> Tuple[Callable, Dict[str, Any]]
ControllerFactory = Callable[[Array], Tuple[Callable, Dict[str, Any]]]
FrictionFactory = Callable[[Array], Tuple[Callable, Dict[str, Any]]]
ChangerFactory = Callable[[Array], Tuple[Callable, Dict[str, Any]]]
InitSpec = Distribution | float


def _sample_init(
    spec: InitSpec,
    seed: Array,
    *,
    shape: Tuple[int, ...] = (),
) -> Array:
    """Sample init value from spec.

    Supported spec formats:
    - scalar number: fixed value
    - Uniform(min, max): uniform(min, max)
    - Normal(mean, std): normal(mean, std)
    """
    min_val = max_val = mean_val = std_val = None

    min_val = getattr(spec, "min", None)
    max_val = getattr(spec, "max", None)
    mean_val = getattr(spec, "mean", None)
    std_val = getattr(spec, "std", None)

    if min_val is not None and max_val is not None:
        return jax.random.uniform(
            seed,
            minval=float(min_val),
            maxval=float(max_val),
            shape=shape,
        )

    if mean_val is not None and std_val is not None:
        return jax.random.normal(seed, shape=shape) * float(std_val) + float(mean_val)

    if isinstance(spec, (int, float)):
        if shape:
            return jnp.full(shape, float(spec))
        return jnp.asarray(float(spec))

    raise ValueError(
        "Unsupported init spec. Use scalar, Uniform(min,max), or Normal(mean,std)."
    )


def init_mlp_params(
    layer_sizes: List[int],
    key: Array,
) -> List[Dict[str, Array]]:
    """Xavier‑initialised weights for a fully‑connected tanh MLP."""
    keys = jax.random.split(key, len(layer_sizes) - 1)
    params: List[Dict[str, Array]] = []
    for k, (in_f, out_f) in zip(keys, zip(layer_sizes[:-1], layer_sizes[1:])):
        w_key, _ = jax.random.split(k)
        w = (
            jax.random.normal(w_key, (in_f, out_f))
            * jnp.sqrt(1.0 / in_f)
        )  # Glorot
        b = jnp.zeros(out_f)
        params.append({"w": w, "b": b})
    return params


def mlp_forward(params: List[Dict[str, Array]], x: Array) -> Array:
    """Forward pass that works for scalar or batched inputs."""
    x = jnp.atleast_2d(x)  # (B, in_features)
    for layer in params[:-1]:
        x = jax.nn.relu(x @ layer["w"] + layer["b"])
    x = x @ params[-1]["w"] + params[-1]["b"]  # (B, 1)
    return x.squeeze(-1)  # → (B,) or scalar


# -----------------------------------------------------------------------------
# Controllers
# -----------------------------------------------------------------------------

# Controllers
def get_controller_pd_fixed(
    kp: float,
    kv: float,
) -> ControllerFactory:
    """Factory maker for a fixed‑gain PD controller."""

    def factory(seed: Array) -> Tuple[Callable, Dict[str, Any]]:
        @jax.jit
        def make_tau(
            params: Dict[str, Any],
            data: mjx.Data,
            ctrl: Array,
            time_index: Array,
        ) -> Array:
            del params, time_index  # unused
            return kp * (ctrl - data.qpos) - kv * data.qvel

        return make_tau, {}

    return factory


def get_controller_pd_variable(
    kp_init: InitSpec,
    kv_init: InitSpec,
) -> ControllerFactory:
    """Factory maker for a learnable PD controller (kp, kv in params).
    Initial kp, kv are sampled from seed using the provided init specs."""

    def factory(seed: Array) -> Tuple[Callable, Dict[str, Any]]:
        k1, k2 = jax.random.split(seed, 2)
        kp_val = _sample_init(kp_init, k1, shape=())
        kv_val = _sample_init(kv_init, k2, shape=())
        init_params = {
            "kp": kp_val,
            "kv": kv_val,
        }

        @jax.jit
        def make_tau(
            params: Dict[str, Any],
            data: mjx.Data,
            ctrl: Array,
            time_index: Array,
        ) -> Array:
            del time_index  # unused
            kp_val = params["kp"]
            kv_val = params["kv"]
            return kp_val * (ctrl - data.qpos) - kv_val * data.qvel

        return make_tau, init_params

    return factory


def get_controller_nn(dims: List[int]) -> ControllerFactory:
    """Factory maker for an MLP-based neural network controller.

    Args:
        dims: List of layer dimensions
              [input_dim, hidden1, hidden2, ..., output_dim]
              Input is [qpos, qvel, ctrl] concatenated.
              Output is torque (same dimension as qpos).

    Returns:
        ControllerFactory that creates an MLP controller.
    """
    if len(dims) < 2:
        raise ValueError(
            "dims must have at least 2 elements [input_dim, output_dim]"
        )

    def factory(seed: Array) -> Tuple[Callable, Dict[str, Any]]:
        mlp_params = init_mlp_params(dims, seed)
        init_params = {"mlp": mlp_params}

        @jax.jit
        def make_tau(
            params: Dict[str, Any],
            data: mjx.Data,
            ctrl: Array,
            time_index: Array,
        ) -> Array:
            del time_index  # unused
            # Concatenate [qpos, qvel, ctrl] as input
            nn_in = jnp.concatenate([data.qpos, data.qvel, ctrl])
            # MLP outputs torque
            return mlp_forward(params["mlp"], nn_in)

        return make_tau, init_params

    return factory


def get_controller_optimal_torque(optimal_torques: Array) -> ControllerFactory:
    """Factory maker for an optimal torque lookup controller.

    Uses explicit time_index parameter to lookup optimal torque.

    Args:
        optimal_torques: Array of shape (T, qpos_size) containing
                         optimal torques for T states.

    Returns:
        ControllerFactory that returns optimal torques via index lookup.
        The make_tau function signature is (params, data, ctrl, time_index).
    """
    optimal_torques = jnp.asarray(optimal_torques, dtype=jnp.float64)

    def factory(seed: Array) -> Tuple[Callable, Dict[str, Any]]:
        init_params = {
            "optimal_torques": optimal_torques,
        }

        @jax.jit
        def make_tau(
            params: Dict[str, Any],
            data: mjx.Data,
            ctrl: Array,
            time_index: Array,
        ) -> Array:
            del ctrl  # unused
            # time_index is the time step index (scalar from scan)
            idx = jnp.asarray(time_index).astype(jnp.int32)

            # Clip to valid range
            # idx = jnp.clip(
            #     idx, 0, params["optimal_torques"].shape[0] - 1
            # )

            return params["optimal_torques"][idx]

        return make_tau, init_params

    return factory


# Frictions
def make_friction_karnopp(
    friction_init: InitSpec,
) -> FrictionFactory:
    def factory(seed: Array) -> Tuple[Callable, Dict[str, Any]]:
        init_params = {
            "frictionKarnopp": _sample_init(friction_init, seed, shape=(1,)),
        }

        @jax.jit
        def friction(
            params: Dict[str, Any],
            data: mjx.Data,
            ctrl: Array,
            tau: Array,
        ) -> Array:
            del ctrl, tau  # unused
            return jnp.where(
                jnp.abs(data.qvel) < 1e-3,
                0.0,
                jnp.sign(data.qvel) * jnp.abs(params["frictionKarnopp"]),
            )

        return friction, init_params

    return factory


def make_friction_smooth_karnopp(
    friction_init: InitSpec,
    smooth_init: InitSpec,
) -> FrictionFactory:
    def factory(seed: Array) -> Tuple[Callable, Dict[str, Any]]:
        keys = jax.random.split(seed, 2)
        init_params = {
            "frictionKarnopp": _sample_init(friction_init, keys[0], shape=(1,)),
            "smoothKarnopp": _sample_init(smooth_init, keys[1], shape=(1,)),
        }

        @jax.jit
        def friction(
            params: Dict[str, Any],
            data: mjx.Data,
            ctrl: Array,
            tau: Array,
        ) -> Array:
            del ctrl, tau  # unused
            gain = params["smoothKarnopp"]
            return jnp.tanh(gain * data.qvel * 10) * jnp.abs(
                params["frictionKarnopp"]
            )

        return friction, init_params

    return factory


def make_friction_karnopp_stribeck(
    coulomb_init: InitSpec,
    static_init: InitSpec,
    stribeck_vel_init: InitSpec,
) -> FrictionFactory:
    def factory(seed: Array) -> Tuple[Callable, Dict[str, Any]]:
        keys = jax.random.split(seed, 3)
        init_params = {
            "frictionCoulomb": _sample_init(coulomb_init, keys[0], shape=(1,)),
            "frictionStatic": _sample_init(static_init, keys[1], shape=(1,)),
            "frictionStribeckVel": _sample_init(stribeck_vel_init, keys[2], shape=(1,)),
        }

        @jax.jit
        def friction(
            params: Dict[str, Any],
            data: mjx.Data,
            ctrl: Array,
            tau: Array,
        ) -> Array:
            del ctrl, tau  # unused
            v = data.qvel
            Fc = jnp.abs(params["frictionCoulomb"])
            Fs = jnp.abs(params["frictionStatic"])
            vs = jnp.abs(params["frictionStribeckVel"])
            stribeck = Fc + (Fs - Fc) * jnp.exp(-jnp.abs(v) / vs)
            return jnp.where(jnp.abs(v) < 1e-3, 0.0, jnp.sign(v) * stribeck)

        return friction, init_params

    return factory


def make_friction_nn(dims: List[int]) -> FrictionFactory:
    """Factory maker for a neural friction term via an MLP.

    Args:
        dims: List of layer dimensions
              [input_dim, hidden1, hidden2, ..., output_dim]
              Input is [qpos, tau, qvel, qfrc_bias] concatenated (4 dims).
              Output is friction scalar (1 dim).

    Returns:
        FrictionFactory that creates an MLP friction model.
    """
    if len(dims) < 2:
        raise ValueError(
            "dims must have at least 2 elements [input_dim, output_dim]"
        )

    def factory(seed: Array) -> Tuple[Callable, Dict[str, Any]]:
        mlp_params = init_mlp_params(dims, seed)
        init_params = {"mlp": mlp_params}

        @jax.jit
        def friction(
            params: Dict[str, Any],
            data: mjx.Data,
            ctrl: Array,
            tau: Array,
        ) -> Array:
            del ctrl  # unused
            # Concatenate [qpos, tau, qvel, qfrc_bias] as input
            nn_in = jnp.concatenate(
                [data.qpos, tau, data.qvel, data.qfrc_bias]
            )
            return mlp_forward(params["mlp"], nn_in)

        return friction, init_params

    return factory


# Changers
def make_changer_armature(
    armature_init: InitSpec,
) -> ChangerFactory:
    def factory(seed: Array) -> Tuple[Callable, Dict[str, Any]]:
        init = {
            "armature": _sample_init(armature_init, seed, shape=(1,))
        }

        @jax.jit
        def change_model(params: Dict, old_model: mjx.Model) -> mjx.Model:
            return old_model.replace(
                dof_armature=abs(params["armature"]),
            )

        return change_model, init

    return factory


def make_changer_damping(
    damping_init: InitSpec,
) -> ChangerFactory:
    def factory(seed: Array) -> Tuple[Callable, Dict[str, Any]]:
        init = {
            "damping": _sample_init(damping_init, seed, shape=(1,)),
        }

        @jax.jit
        def change_model(params: Dict, old_model: mjx.Model) -> mjx.Model:
            return old_model.replace(
                dof_damping=abs(params["damping"]),
            )

        return change_model, init

    return factory


def make_changer_frictionloss(
    frictionloss_init: InitSpec,
) -> ChangerFactory:
    def factory(seed: Array) -> Tuple[Callable, Dict[str, Any]]:
        init = {
            "frictionloss": _sample_init(frictionloss_init, seed, shape=(1,)),
        }

        @jax.jit
        def change_model(params: Dict, old_model: mjx.Model) -> mjx.Model:
            return old_model.replace(
                dof_frictionloss=abs(params["frictionloss"]),
            )

        return change_model, init

    return factory


def compose_action_model(
    controllers: List[ControllerFactory],
    frictions: List[FrictionFactory],
    changers: List[ChangerFactory],
    seed: Array,
    controller_names: Optional[List[str]] = None,
    friction_names: Optional[List[str]] = None,
    changer_names: Optional[List[str]] = None,
) -> Tuple[Callable, Callable, Dict[str, Any]]:
    """Compose controllers/frictions/changers lists into one action/model pair."""
    n_keys = len(controllers) + len(frictions) + len(changers)
    keys = jax.random.split(seed, n_keys)

    controller_makes: List[Callable] = []
    controller_params: List[Dict[str, Any]] = []
    for fac, k in zip(controllers, keys[:len(controllers)]):
        make_i, p_i = fac(k)
        controller_makes.append(make_i)
        controller_params.append(p_i)

    friction_makes: List[Callable] = []
    friction_params: List[Dict[str, Any]] = []
    fric_start = len(controllers)
    for fac, k in zip(frictions, keys[fric_start:fric_start + len(frictions)]):
        make_i, p_i = fac(k)
        friction_makes.append(make_i)
        friction_params.append(p_i)

    changer_makes: List[Callable] = []
    changer_params: List[Dict[str, Any]] = []
    chg_start = fric_start + len(frictions)
    for fac, k in zip(changers, keys[chg_start:]):
        make_i, p_i = fac(k)
        changer_makes.append(make_i)
        changer_params.append(p_i)

    if controller_names is None:
        controller_names = [f"ctrl_{i}" for i in range(len(controllers))]
    if friction_names is None:
        friction_names = [f"fric_{i}" for i in range(len(frictions))]
    if changer_names is None:
        changer_names = [f"chg_{i}" for i in range(len(changers))]

    init_params = {
        "controllers": {n: p for n, p in zip(controller_names, controller_params)},
        "frictions": {n: p for n, p in zip(friction_names, friction_params)},
        "changers": {n: p for n, p in zip(changer_names, changer_params)},
    }

    @jax.jit
    def make_action(
        params: Dict[str, Any],
        data: mjx.Data,
        ctrl: Array,
        time_index: Array,
    ) -> Array:
        tau = jnp.zeros_like(ctrl)
        for name, make in zip(controller_names, controller_makes):
            tau = tau + make(params["controllers"][name], data, ctrl, time_index)

        fric_total = jnp.zeros_like(tau)
        for name, make in zip(friction_names, friction_makes):
            fric_total = fric_total + make(
                params["frictions"][name],
                data,
                ctrl,
                tau,
            )
        return tau - fric_total

    @jax.jit
    def change_model(
        params: Dict[str, Any],
        old_model: mjx.Model,
    ) -> mjx.Model:
        model = old_model
        for name, change in zip(changer_names, changer_makes):
            model = change(params["changers"][name], model)
        return model

    return make_action, change_model, init_params
