import jax
import jax.numpy as jnp
from jax import Array
from typing import List, Dict, Any, Tuple, Callable, NamedTuple, Sequence
from typing import Optional
from mujoco import mjx
from sysidmjx.distributions import Distribution

# Each factory has signature: def factory(seed: Array) -> Tuple[Callable, Dict[str, Any]]
ModelFactory = Callable[[Array], Tuple[Callable, Dict[str, Any]]]
FrictionFactory = Callable[[Array], Tuple[Callable, Dict[str, Any]]]
ChangerFactory = Callable[[Array], Tuple[Callable, Dict[str, Any]]]
InitSpec = Distribution | float | Sequence[float]


class ModelState(NamedTuple):
    qpos_hist: Array
    qvel_hist: Array
    qact_hist: Array
    time_index: Array
    control: Dict[str, Array]


def _sample_init(
    spec: InitSpec,
    seed: Array,
    *,
    shape: Tuple[int, ...] = (),
) -> Array:
    """Sample init value from spec.

    Supported spec formats:
    - scalar number: fixed value
    - sequence: fixed vector value
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

    if isinstance(spec, Sequence):
        values = jnp.asarray(spec, dtype=float)
        return jnp.broadcast_to(values, shape) if shape else values

    raise ValueError(
        "Unsupported init spec. Use scalar, sequence, Uniform(min,max), or Normal(mean,std)."
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
# Model chain components
# -----------------------------------------------------------------------------


def get_model_action_passthrough() -> ModelFactory:
    """Leave qact unchanged so the dataset/control input is used directly."""

    def factory(seed: Array) -> Tuple[Callable, Dict[str, Any]]:
        del seed

        @jax.jit
        def model(
            params: Dict[str, Any],
            data: mjx.Data,
            state: ModelState,
        ) -> ModelState:
            del params, data
            return state

        return model, {}

    return factory

def get_model_pd_fixed(
    kp: float,
    kv: float,
) -> ModelFactory:
    """Fixed-gain PD transform over the whole virtual history."""

    def factory(seed: Array) -> Tuple[Callable, Dict[str, Any]]:
        del seed

        @jax.jit
        def model(
            params: Dict[str, Any],
            data: mjx.Data,
            state: ModelState,
        ) -> ModelState:
            del params, data
            qact_hist = (
                float(kp) * (state.qact_hist - state.qpos_hist)
                - float(kv) * state.qvel_hist
            )
            return state._replace(qact_hist=qact_hist)

        return model, {}

    return factory


def get_model_torque_constant_from_qvel(
    torque_const_init: InitSpec,
    torque_const_vel_init: InitSpec,
) -> ModelFactory:
    """Learned current-to-torque scale for the current virtual action.

    qact_out = qact_in * (c0 + abs(qvel) * c1)
    """

    def factory(seed: Array) -> Tuple[Callable, Dict[str, Any]]:
        k1, k2 = jax.random.split(seed, 2)
        init_params = {
            "torque_const": _sample_init(torque_const_init, k1, shape=(1,)),
            "torque_const_vel": _sample_init(torque_const_vel_init, k2, shape=(1,)),
        }

        @jax.jit
        def model(
            params: Dict[str, Any],
            data: mjx.Data,
            state: ModelState,
        ) -> ModelState:
            del data
            torque_const = jnp.abs(params["torque_const"]) + (
                jnp.abs(state.qvel_hist) * jnp.abs(params["torque_const_vel"])
            )
            return state._replace(qact_hist=state.qact_hist * torque_const)

        return model, init_params

    return factory


def get_model_pd_variable(
    kp_init: InitSpec,
    kv_init: InitSpec,
) -> ModelFactory:
    """Learnable PD transform over the whole virtual history (kp, kv in params).
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
        def model(
            params: Dict[str, Any],
            data: mjx.Data,
            state: ModelState,
        ) -> ModelState:
            del data
            kp_val = params["kp"]
            kv_val = params["kv"]
            qact_hist = (
                kp_val * (state.qact_hist - state.qpos_hist)
                - kv_val * state.qvel_hist
            )
            return state._replace(qact_hist=qact_hist)

        return model, init_params

    return factory


def get_model_nn(
    hidden_dims: List[int],
    history_len: int,
    torque_dim: int = 1,
) -> ModelFactory:
    """MLP transform that writes the current virtual action from full history.

    Args:
        hidden_dims: Hidden layer dimensions.
        history_len: Number of qpos/qvel/qact history samples.
        torque_dim: Output torque dimension.

    Returns:
        ModelFactory that creates an MLP state transform.
    """
    dims = [3 * history_len, *hidden_dims, torque_dim]

    def factory(seed: Array) -> Tuple[Callable, Dict[str, Any]]:
        mlp_params = init_mlp_params(dims, seed)
        init_params = {"mlp": mlp_params}

        @jax.jit
        def model(
            params: Dict[str, Any],
            data: mjx.Data,
            state: ModelState,
        ) -> ModelState:
            del data
            nn_in = jnp.concatenate(
                [
                    jnp.ravel(state.qpos_hist),
                    jnp.ravel(state.qvel_hist),
                    jnp.ravel(state.qact_hist),
                ]
            )
            qact = mlp_forward(params["mlp"], nn_in)
            return state._replace(qact_hist=state.qact_hist.at[-1].set(qact))

        return model, init_params

    return factory


def get_model_nn_torque_residual(
    hidden_dims: List[int],
    history_len: int,
    torque_dim: int = 1,
    correction_limit: float = 1.0,
) -> ModelFactory:
    """Current qact plus learned correction from full virtual history."""
    dims = [3 * history_len, *hidden_dims, torque_dim]

    def factory(seed: Array) -> Tuple[Callable, Dict[str, Any]]:
        mlp_params = init_mlp_params(dims, seed)
        mlp_params[-1] = {
            "w": jnp.zeros_like(mlp_params[-1]["w"]),
            "b": jnp.zeros_like(mlp_params[-1]["b"]),
        }
        init_params = {"mlp": mlp_params}

        @jax.jit
        def model(
            params: Dict[str, Any],
            data: mjx.Data,
            state: ModelState,
        ) -> ModelState:
            del data
            nn_in = jnp.concatenate(
                [
                    jnp.ravel(state.qpos_hist),
                    jnp.ravel(state.qvel_hist),
                    jnp.ravel(state.qact_hist),
                ]
            )
            delta_tau = float(correction_limit) * jnp.tanh(
                mlp_forward(params["mlp"], nn_in)
            )
            qact = state.qact_hist[-1] + delta_tau
            return state._replace(qact_hist=state.qact_hist.at[-1].set(qact))

        return model, init_params

    return factory


def get_model_nn_qvel_qact_residual(
    hidden_dims: List[int],
    history_len: int,
    torque_dim: int = 1,
) -> ModelFactory:
    """Current qact plus learned residual from virtual qact/qvel history."""
    dims = [2 * history_len, *hidden_dims, torque_dim]

    def factory(seed: Array) -> Tuple[Callable, Dict[str, Any]]:
        mlp_params = init_mlp_params(dims, seed)
        mlp_params[-1] = {
            "w": jnp.zeros_like(mlp_params[-1]["w"]),
            "b": jnp.zeros_like(mlp_params[-1]["b"]),
        }
        init_params = {"mlp": mlp_params}

        @jax.jit
        def model(
            params: Dict[str, Any],
            data: mjx.Data,
            state: ModelState,
        ) -> ModelState:
            del data
            qact = state.qact_hist[-1]
            delta_tau = mlp_forward(
                params["mlp"],
                jnp.concatenate(
                    [jnp.ravel(state.qact_hist), jnp.ravel(state.qvel_hist)]
                ),
            )
            qact_next = qact + delta_tau
            return state._replace(qact_hist=state.qact_hist.at[-1].set(qact_next))

        return model, init_params

    return factory


def get_model_nn_qvel_qact_residual_normalized(
    hidden_dims: List[int],
    history_len: int,
    torque_dim: int = 1,
    qvel_scale: float = 12.0,
    qact_scale: float = 4.0,
) -> ModelFactory:
    """Current qact plus scaled learned residual from normalized qact/qvel history.

    The NN sees normalized qact/qvel histories and predicts residual torque in
    qact-scale units. Last layer is zero-initialized, so initial policy is
    exactly torque passthrough.
    """
    dims = [2 * history_len, *hidden_dims, torque_dim]

    def factory(seed: Array) -> Tuple[Callable, Dict[str, Any]]:
        mlp_params = init_mlp_params(dims, seed)
        mlp_params[-1] = {
            "w": jnp.zeros_like(mlp_params[-1]["w"]),
            "b": jnp.zeros_like(mlp_params[-1]["b"]),
        }
        init_params = {"mlp": mlp_params}

        @jax.jit
        def model(
            params: Dict[str, Any],
            data: mjx.Data,
            state: ModelState,
        ) -> ModelState:
            del data
            qact = state.qact_hist[-1]
            nn_in = jnp.concatenate(
                [
                    jnp.ravel(state.qact_hist) / float(qact_scale),
                    jnp.ravel(state.qvel_hist) / float(qvel_scale),
                ]
            )
            delta_tau = float(qact_scale) * mlp_forward(params["mlp"], nn_in)
            qact_next = qact + delta_tau
            return state._replace(qact_hist=state.qact_hist.at[-1].set(qact_next))

        return model, init_params

    return factory


def get_model_pd_control(
    target_qpos_field: str = "target_qpos",
    target_qvel_field: str = "target_qvel",
    kp_field: str = "kp",
    kd_field: str = "kd",
) -> ModelFactory:
    """Dataset-driven PD torque with no learned action parameters."""

    def factory(seed: Array) -> Tuple[Callable, Dict[str, Any]]:
        del seed

        @jax.jit
        def model(
            params: Dict[str, Any],
            data: mjx.Data,
            state: ModelState,
        ) -> ModelState:
            del params
            target_qpos = state.control[target_qpos_field]
            target_qvel = state.control[target_qvel_field]
            kp = state.control[kp_field]
            kd = state.control[kd_field]
            tau_pd = kp * (target_qpos - data.qpos) + kd * (target_qvel - data.qvel)
            return state._replace(qact_hist=state.qact_hist.at[-1].set(tau_pd))

        return model, {}

    return factory


def get_model_pd_residual(
    hidden_dims: List[int],
    input_signals: Tuple[str, ...] = ("tau_cmd", "qpos", "qvel"),
    history_len: int = 1,
    torque_dim: int = 1,
    target_qpos_field: str = "target_qpos",
    target_qvel_field: str = "target_qvel",
    kp_field: str = "kp",
    kd_field: str = "kd",
) -> ModelFactory:
    """Dataset-driven PD torque plus learned residual.

    Required control fields are configured by DatasetSpec.rollout_fields and
    made available in ModelState.control at each rollout step.
    """
    dims = [len(input_signals) * history_len * torque_dim, *hidden_dims, torque_dim]

    def factory(seed: Array) -> Tuple[Callable, Dict[str, Any]]:
        mlp_params = init_mlp_params(dims, seed)
        mlp_params[-1] = {
            "w": jnp.zeros_like(mlp_params[-1]["w"]),
            "b": jnp.zeros_like(mlp_params[-1]["b"]),
        }
        init_params = {"mlp": mlp_params}

        @jax.jit
        def model(
            params: Dict[str, Any],
            data: mjx.Data,
            state: ModelState,
        ) -> ModelState:
            target_qpos = state.control[target_qpos_field]
            target_qvel = state.control[target_qvel_field]
            kp = state.control[kp_field]
            kd = state.control[kd_field]
            tau_pd = kp * (target_qpos - data.qpos) + kd * (target_qvel - data.qvel)
            tau_pd_hist = kp * (target_qpos - state.qpos_hist[-history_len:]) + kd * (
                target_qvel - state.qvel_hist[-history_len:]
            )

            features = []
            for signal in input_signals:
                if signal == "tau_cmd":
                    features.append(tau_pd_hist)
                elif signal == "logged_tau_cmd":
                    features.append(jnp.broadcast_to(state.control["tau_cmd"], tau_pd_hist.shape))
                elif signal == "tau_pd":
                    features.append(tau_pd_hist)
                elif signal == "qpos":
                    features.append(state.qpos_hist[-history_len:])
                elif signal == "qvel":
                    features.append(state.qvel_hist[-history_len:])
                elif signal == "target_qpos":
                    features.append(jnp.broadcast_to(target_qpos, tau_pd_hist.shape))
                elif signal == "target_qvel":
                    features.append(jnp.broadcast_to(target_qvel, tau_pd_hist.shape))
                else:
                    features.append(jnp.broadcast_to(state.control[signal], tau_pd_hist.shape))

            nn_in = jnp.concatenate([jnp.ravel(feature) for feature in features])
            delta_tau = mlp_forward(params["mlp"], nn_in)
            tau = tau_pd + delta_tau
            return state._replace(qact_hist=state.qact_hist.at[-1].set(tau))

        return model, init_params

    return factory


def get_model_isotropic_pd_residual(
    hidden_dims: List[int],
    input_signals: Tuple[str, ...] = ("tau_cmd", "qpos", "qvel"),
    history_len: int = 1,
    torque_dim: int = 1,
    target_qpos_field: str = "target_qpos",
    target_qvel_field: str = "target_qvel",
    kp_field: str = "kp",
    kd_field: str = "kd",
) -> ModelFactory:
    """Dataset-driven PD torque plus odd-symmetric learned residual.

    The residual is constrained as ``g(x) = f(x) - f(-x)``.
    """
    dims = [len(input_signals) * history_len * torque_dim, *hidden_dims, torque_dim]

    def factory(seed: Array) -> Tuple[Callable, Dict[str, Any]]:
        mlp_params = init_mlp_params(dims, seed)
        mlp_params[-1] = {
            "w": jnp.zeros_like(mlp_params[-1]["w"]),
            "b": jnp.zeros_like(mlp_params[-1]["b"]),
        }
        init_params = {"mlp": mlp_params}

        @jax.jit
        def model(
            params: Dict[str, Any],
            data: mjx.Data,
            state: ModelState,
        ) -> ModelState:
            target_qpos = state.control[target_qpos_field]
            target_qvel = state.control[target_qvel_field]
            kp = state.control[kp_field]
            kd = state.control[kd_field]
            tau_pd = kp * (target_qpos - data.qpos) + kd * (target_qvel - data.qvel)
            tau_pd_hist = kp * (target_qpos - state.qpos_hist[-history_len:]) + kd * (
                target_qvel - state.qvel_hist[-history_len:]
            )

            features = []
            for signal in input_signals:
                if signal == "tau_cmd":
                    features.append(tau_pd_hist)
                elif signal == "logged_tau_cmd":
                    features.append(jnp.broadcast_to(state.control["tau_cmd"], tau_pd_hist.shape))
                elif signal == "tau_pd":
                    features.append(tau_pd_hist)
                elif signal == "qpos":
                    features.append(state.qpos_hist[-history_len:])
                elif signal == "qvel":
                    features.append(state.qvel_hist[-history_len:])
                elif signal == "target_qpos":
                    features.append(jnp.broadcast_to(target_qpos, tau_pd_hist.shape))
                elif signal == "target_qvel":
                    features.append(jnp.broadcast_to(target_qvel, tau_pd_hist.shape))
                else:
                    features.append(jnp.broadcast_to(state.control[signal], tau_pd_hist.shape))

            nn_in = jnp.concatenate([jnp.ravel(feature) for feature in features])
            delta_tau = mlp_forward(params["mlp"], nn_in) - mlp_forward(
                params["mlp"],
                -nn_in,
            )
            tau = tau_pd + delta_tau
            return state._replace(qact_hist=state.qact_hist.at[-1].set(tau))

        return model, init_params

    return factory


def get_model_optimal_torque(optimal_torques: Array) -> ModelFactory:
    """Factory maker for an optimal torque lookup model.

    Uses explicit time_index parameter to lookup optimal torque.

    Args:
        optimal_torques: Array of shape (T, qpos_size) containing
                         optimal torques for T states.

    Returns:
        ModelFactory that writes optimal torques into the current qact slot.
    """
    optimal_torques = jnp.asarray(optimal_torques, dtype=jnp.float64)

    def factory(seed: Array) -> Tuple[Callable, Dict[str, Any]]:
        init_params = {
            "optimal_torques": optimal_torques,
        }

        @jax.jit
        def model(
            params: Dict[str, Any],
            data: mjx.Data,
            state: ModelState,
        ) -> ModelState:
            del data
            # time_index is the time step index (scalar from scan)
            idx = jnp.asarray(state.time_index).astype(jnp.int32)

            # Clip to valid range
            # idx = jnp.clip(
            #     idx, 0, params["optimal_torques"].shape[0] - 1
            # )

            qact = params["optimal_torques"][idx]
            return state._replace(qact_hist=state.qact_hist.at[-1].set(qact))

        return model, init_params

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
            tau: Array,
            state: ModelState,
        ) -> Array:
            del tau, state  # unused
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
            tau: Array,
            state: ModelState,
        ) -> Array:
            del tau, state  # unused
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
            tau: Array,
            state: ModelState,
        ) -> Array:
            del tau, state  # unused
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
            tau: Array,
            state: ModelState,
        ) -> Array:
            del state  # unused
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


def _valid_solimp(values: Array) -> Array:
    values = jnp.ravel(values)
    return jnp.stack((
        jnp.clip(jnp.abs(values[0]), 1e-4, 0.9999),
        jnp.clip(jnp.abs(values[1]), 1e-4, 0.9999),
        jnp.maximum(jnp.abs(values[2]), 1e-9),
        jnp.clip(jnp.abs(values[3]), 1e-4, 0.9999),
        jnp.maximum(jnp.abs(values[4]), 1.0),
    ))


def _valid_solreffriction(values: Array, min_timeconst: float) -> Array:
    values = jnp.ravel(jnp.abs(values))
    return jnp.stack((
        jnp.maximum(values[0], min_timeconst),
        jnp.maximum(values[1], 1e-9),
    ))


def make_changer_solreffriction(
    solreffriction_init: InitSpec,
    min_timeconst: float = 0.002,
    dof_id: int = 0,
) -> ChangerFactory:
    def factory(seed: Array) -> Tuple[Callable, Dict[str, Any]]:
        init = {"solreffriction": _sample_init(solreffriction_init, seed, shape=(2,))}

        @jax.jit
        def change_model(params: Dict, old_model: mjx.Model) -> mjx.Model:
            return old_model.replace(
                dof_solref=old_model.dof_solref.at[dof_id].set(
                    _valid_solreffriction(params["solreffriction"], min_timeconst),
                ),
            )

        return change_model, init

    return factory


def make_changer_solimpfriction(
    solimpfriction_init: InitSpec,
    dof_id: int = 0,
) -> ChangerFactory:
    def factory(seed: Array) -> Tuple[Callable, Dict[str, Any]]:
        init = {"solimpfriction": _sample_init(solimpfriction_init, seed, shape=(5,))}

        @jax.jit
        def change_model(params: Dict, old_model: mjx.Model) -> mjx.Model:
            return old_model.replace(
                dof_solimp=old_model.dof_solimp.at[dof_id].set(
                    _valid_solimp(params["solimpfriction"]),
                ),
            )

        return change_model, init

    return factory


def make_changer_actuator_filter_timeconst(
    timeconst_init: InitSpec,
    actuator_id: int = 0,
) -> ChangerFactory:
    def factory(seed: Array) -> Tuple[Callable, Dict[str, Any]]:
        init = {
            "actuator_filter_timeconst": _sample_init(timeconst_init, seed, shape=(1,)),
        }

        @jax.jit
        def change_model(params: Dict, old_model: mjx.Model) -> mjx.Model:
            timeconst = jnp.ravel(jnp.abs(params["actuator_filter_timeconst"]))[0]
            actuator_dynprm = old_model.actuator_dynprm
            actuator_dynprm = actuator_dynprm.at[actuator_id, 0].set(timeconst)
            return old_model.replace(actuator_dynprm=actuator_dynprm)

        return change_model, init

    return factory


def make_changer_actuator_gain(
    gain_init: InitSpec,
    actuator_id: int = 0,
) -> ChangerFactory:
    def factory(seed: Array) -> Tuple[Callable, Dict[str, Any]]:
        init = {
            "actuator_gain": _sample_init(gain_init, seed, shape=(1,)),
        }

        @jax.jit
        def change_model(params: Dict, old_model: mjx.Model) -> mjx.Model:
            gain = jnp.ravel(jnp.abs(params["actuator_gain"]))[0]
            actuator_gainprm = old_model.actuator_gainprm
            actuator_gainprm = actuator_gainprm.at[actuator_id, 0].set(gain)
            return old_model.replace(actuator_gainprm=actuator_gainprm)

        return change_model, init

    return factory


def compose_action_model(
    models: List[ModelFactory],
    frictions: List[FrictionFactory],
    changers: List[ChangerFactory],
    seed: Array,
    model_names: Optional[List[str]] = None,
    friction_names: Optional[List[str]] = None,
    changer_names: Optional[List[str]] = None,
) -> Tuple[Callable, Callable, Dict[str, Any]]:
    """Compose state-transform models, frictions, and changers."""
    n_keys = len(models) + len(frictions) + len(changers)
    keys = jax.random.split(seed, n_keys)

    model_makes: List[Callable] = []
    model_params: List[Dict[str, Any]] = []
    for fac, k in zip(models, keys[:len(models)]):
        make_i, p_i = fac(k)
        model_makes.append(make_i)
        model_params.append(p_i)

    friction_makes: List[Callable] = []
    friction_params: List[Dict[str, Any]] = []
    fric_start = len(models)
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

    if model_names is None:
        model_names = [f"model_{i}" for i in range(len(models))]
    if friction_names is None:
        friction_names = [f"fric_{i}" for i in range(len(frictions))]
    if changer_names is None:
        changer_names = [f"chg_{i}" for i in range(len(changers))]

    init_params = {
        "models": {n: p for n, p in zip(model_names, model_params)},
        "frictions": {n: p for n, p in zip(friction_names, friction_params)},
        "changers": {n: p for n, p in zip(changer_names, changer_params)},
    }

    @jax.jit
    def make_action(
        params: Dict[str, Any],
        data: mjx.Data,
        state: ModelState,
    ) -> Array:
        for name, make in zip(model_names, model_makes):
            state = make(params["models"][name], data, state)

        tau = state.qact_hist[-1]
        fric_total = jnp.zeros_like(tau)
        for name, make in zip(friction_names, friction_makes):
            fric_total = fric_total + make(
                params["frictions"][name],
                data,
                tau,
                state,
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
