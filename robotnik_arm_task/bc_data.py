"""Data side of behaviour cloning: model inputs/outputs, loading saved episodes, and the episode split.

make_features is shared by training (whole episodes) and robotnik_arm_policy_task.py (one frame per step),
so the model always sees inputs built the same way.
"""
import numpy as np
from collections import deque
from functools import cache
from pathlib import Path


def to_base_frame(v_xy, base_bearing):
    """Horizontal world-frame vectors (..., 2) expressed in the base's frame (x forward, y left)."""
    c, s = np.cos(base_bearing), np.sin(base_bearing)
    return np.stack([c * v_xy[..., 0] + s * v_xy[..., 1], -s * v_xy[..., 0] + c * v_xy[..., 1]], axis=-1)


def make_features(base_pos, base_bearing, q_arm, qd_arm, target, ee_pos):
    """Model input in the robot's frame (PLAN.md step 6). Works for one frame (1-D arrays) or a whole episode.

    No absolute base pose, so the same situation gives the same inputs wherever the cart is.
    """
    base_bearing = np.asarray(base_bearing)
    target = np.broadcast_to(target, base_pos.shape)           # one target, copied to every frame
    return np.concatenate([
        to_base_frame(target[..., :2] - base_pos[..., :2], base_bearing),   # target forward, left of the base (2)
        target[..., 2:3],                          # target height: the floor is flat, so it means the same anywhere (1)
        to_base_frame(target[..., :2] - ee_pos[..., :2], base_bearing),     # target relative to the EE, base frame (2)
        target[..., 2:3] - ee_pos[..., 2:3],       # ... and its height above the EE (1)
        # Arm joint angles as sin/cos: J1, J3, J5, J7 have no limits and could wrap past +-pi (in data_main they stay
        # within +-0.7 rad, so this is for safety, e.g. randomised arm start poses, PLAN.md step 20); all seven for
        # simplicity                                                                        (14)
        np.sin(q_arm),
        np.cos(q_arm),
        qd_arm,                                    # arm joint velocities    (7)
    ], axis=-1).astype(np.float32)                 # 27 inputs


# Columns of make_features each split network uses (PLAN.md step 15). The base only needs where the target is: the
# expert's wheel commands don't depend on the arm. The arm doesn't need the target relative to the base: the target
# relative to the shoulder follows from the EE-relative target and the joint angles.
BASE_INPUTS = slice(0, 3)    # target forward, left of the base, target height
ARM_INPUTS = slice(3, 27)    # target relative to the EE, sin/cos of the joint angles, joint velocities


# Base wheel geometry the expert uses to turn (v, w) into wheel commands (PLAN.md step 13): the values of
# BaseControllerParams in robotnik_arm_control.py, keep the two in step
WHEEL_RADIUS = 0.11       # m
WHEEL_BASE_WIDTH = 0.66   # m, effective track width for skid-steer (calibrated in simulation)


def wheels_to_vw(wheel_cmds):
    """4 wheel commands [front_right, front_left, back_left, back_right] (rad/s) to forward speed v (m/s) and turn
    rate w (rad/s), (..., 2). Exact for the expert's commands: each side's two wheels always get the same command."""
    right, left = wheel_cmds[..., 0], wheel_cmds[..., 1]
    return np.stack([WHEEL_RADIUS * (right + left) / 2,
                     WHEEL_RADIUS * (right - left) / WHEEL_BASE_WIDTH], axis=-1)


def vw_to_wheels(vw):
    """Inverse of wheels_to_vw: the expert's own formula (base_controller in robotnik_arm_control.py)."""
    v, w = vw[..., 0], vw[..., 1]
    right = (v + w * WHEEL_BASE_WIDTH / 2) / WHEEL_RADIUS
    left = (v - w * WHEEL_BASE_WIDTH / 2) / WHEEL_RADIUS
    return np.stack([right, left, left, right], axis=-1)


def make_labels(wheel_cmds, torque_arm, vw_outputs=False):
    """Model output: 4 wheel commands (or v, w with vw_outputs, PLAN.md step 13) then 7 arm torques."""
    wheels = wheels_to_vw(wheel_cmds) if vw_outputs else wheel_cmds
    return np.concatenate([wheels, torque_arm], axis=-1).astype(np.float32)


def decode_outputs(y, tau_g=None, vw_outputs=False):
    """Model output back to robot commands: (4 wheel commands in rad/s, 7 arm torques in N*m).

    The inverse of make_labels. Works for one frame or a whole episode (one row per frame).
    Everything that runs or scores a model (policy script, offline test) goes through this, so a new output
    format (e.g. (v, w) instead of 4 wheel speeds) only changes make_labels and this function.
    tau_g: for gravity-residual models (PLAN.md step 12) the arm gravity torques of the same frames
    (arm_gravity), added back to the predicted torques; None for models that predict the full torque.
    vw_outputs: the model outputs (v, w) instead of 4 wheel commands (PLAN.md step 13).
    """
    y = np.asarray(y)
    wheels = vw_to_wheels(y[..., :2]) if vw_outputs else y[..., :4]
    torques = y[..., -7:]   # always the last 7, whichever wheel format
    if tau_g is not None:
        torques = torques + tau_g
    return wheels, torques


@cache
def _gravity_model():
    """Pinocchio model of the robot and the joint indices arm_gravity needs; built once, on the first call."""
    import pinocchio as pin   # only needed here, so the rest of the module keeps working without it
    urdf = Path(__file__).resolve().parent / "models" / "summit_xl_arm_description" / "summit_xl_complete_pin.urdf"
    model = pin.buildModelFromUrdf(str(urdf), pin.JointModelFreeFlyer())
    arm = [f"robot_j2s7s300_joint_{i}" for i in range(1, 8)]
    fingers = [f"robot_j2s7s300_joint_finger{s}" for s in ("_1", "_tip_1", "_2", "_tip_2", "_3", "_tip_3")]
    # Pinocchio orders the joints its own way: look up where each arm/finger joint sits by name
    idx_q = [model.joints[model.getJointId(n)].idx_q for n in arm + fingers]
    idx_v = [model.joints[model.getJointId(n)].idx_v for n in arm]
    return model, model.createData(), idx_q, idx_v


def arm_gravity(q_mes):
    """Arm gravity torques in N*m, (n, 7), from the measured joint angles q_mes, (n, 17) or one frame (17,).

    The same torque the expert adds in arm_controller (pin.computeGeneralizedGravity), but built from the URDF alone,
    so it needs no simulator and works on saved episodes (PLAN.md step 12). The base is left at the origin and
    upright: on a flat floor the base position and yaw don't change the arm's gravity torques. Checked on data_pilot:
    matches the expert's torques to ~0.006 N*m (the small rest is the base tilting while it drives).
    """
    import pinocchio as pin
    model, data, idx_q, idx_v = _gravity_model()

    q_mes = np.atleast_2d(q_mes)
    q = pin.neutral(model)
    tau_g = np.empty((len(q_mes), 7), dtype=np.float32)
    for k, qm in enumerate(q_mes):
        q[idx_q] = qm[4:17]   # arm (7) then fingers (6), same order as q_mes
        tau_g[k] = pin.computeGeneralizedGravity(model, data, q)[idx_v]
    return tau_g


def load_episode(path, gravity_residual=False, vw_outputs=False):
    """Inputs and labels for every frame of one saved episode.

    gravity_residual: the torque labels are the expert torque minus the arm gravity torque (PLAN.md step 12).
    vw_outputs: the wheel labels are (v, w) instead of 4 wheel commands (PLAN.md step 13).
    """
    d = np.load(path)
    x = make_features(d["base_pos"], d["base_bearing"], d["q_mes"][:, 4:11], d["qd_mes"][:, 4:11], d["target"],
                      d["ee_pos"])
    torque = d["torque_arm"] - arm_gravity(d["q_mes"]) if gravity_residual else d["torque_arm"]
    y = make_labels(d["wheel_cmds"], torque, vw_outputs)
    return x, y


def subsample_indices(wheel_cmds, torque_arm, keep_frac=0.1, mode="adaptive", floor=0.3, smooth=50, seed=0):
    """Frame indices to keep from one episode (sorted, no repeats).

    "uniform": evenly spaced frames. "adaptive": frames drawn at random, more of them where the expert's commands
    change (phase switches, the reach). A frame's chance of being kept is floor * uniform + (1 - floor) * change
    score, so a share `floor` of the kept frames is spread over the whole episode and no part is left out.
    Wheels and torques get half of the change score each, so the torques (which change far more often) don't drown
    out the few wheel switches. The change is averaged over `smooth` frames, so the frames around a jump count too.
    seed can be a tuple, e.g. (seed, episode id), so every episode gets its own repeatable draw.
    """
    n = len(wheel_cmds)
    k = max(1, min(n, round(n * keep_frac)))
    if mode == "uniform":
        return np.linspace(0, n - 1, k).round().astype(int)
    if mode != "adaptive":
        raise ValueError(f"unknown subsampling mode {mode!r}")

    p = np.full(n, floor / n)
    kernel = np.ones(min(smooth, n)) / min(smooth, n)
    for cmds in (wheel_cmds, torque_arm):
        change = np.r_[0.0, np.abs(np.diff(cmds, axis=0)).sum(axis=1)]
        change = np.convolve(change, kernel, mode="same")
        share = change / change.sum() if change.sum() > 0 else np.full(n, 1.0 / n)
        p += (1 - floor) / 2 * share
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=k, replace=False, p=p / p.sum()))


def with_history(x, frames, history_k=0, history_spacing=1):
    """Model inputs of the given frames of one episode, each with its history_k past frames (PLAN.md step 14).

    x: the inputs of every frame of the episode, (n, 27). Returns (len(frames), 27 * (history_k + 1)): the current
    frame first, then t - spacing, t - 2 * spacing, ... (spacing in frames, 1 frame = 1 ms). History comes from the
    full episode, so the past frames don't have to be kept by subsampling. Before the first frame there is no past:
    that frame is repeated (not zeros, which would mean "an average state" after normalisation). FrameHistory does
    the same at run time. history_k=0 gives just the frames themselves.
    """
    offsets = history_spacing * np.arange(history_k + 1)
    idx = np.maximum(np.asarray(frames)[:, None] - offsets, 0)
    return x[idx].reshape(len(idx), -1)


class FrameHistory:
    """Run-time version of with_history: keeps the recent input rows, one new row per simulation step.

    Create one per episode. The buffer starts filled with the first row, as with_history repeats the first frame.
    """

    def __init__(self, history_k=0, history_spacing=1):
        self.offsets = history_spacing * np.arange(history_k + 1)
        self.rows = deque(maxlen=history_k * history_spacing + 1)

    def add(self, x):
        """Store this step's input row and return it joined with its past frames."""
        if not self.rows:
            self.rows.extend([x] * self.rows.maxlen)
        self.rows.append(x)
        return np.concatenate([self.rows[-1 - o] for o in self.offsets])


def load_episodes(data_dir, episode_ids, keep_frac=None, mode="adaptive", seed=0, gravity_residual=False,
                  arrived=False, vw_outputs=False, history_k=0, history_spacing=1):
    """Inputs and labels of several episodes stacked into one array each.

    keep_frac=None keeps every frame; otherwise each episode keeps only the frames picked by subsample_indices.
    gravity_residual: torque labels minus the arm gravity torque (see load_episode).
    arrived: also return the expert's "arrived" flag (nav_done, 0/1) of the same frames, as a third array, for the
    split networks (PLAN.md step 15).
    vw_outputs: wheel labels as (v, w) (see load_episode).
    history_k, history_spacing: past frames added to each input (see with_history).
    """
    xs, ys, arrs = [], [], []
    for i in episode_ids:
        path = Path(data_dir) / f"ep_{i:04d}.npz"
        x, y = load_episode(path)
        keep = np.arange(len(x))
        if keep_frac is not None:
            # Picked from the full torques, so a gravity-residual model trains on the same frames as one without
            keep = subsample_indices(*decode_outputs(y), keep_frac=keep_frac, mode=mode, seed=(seed, i))
        # Before dropping any frames: the history comes from the full episode
        x, y = with_history(x, keep, history_k, history_spacing), y[keep]
        if gravity_residual:
            y[:, 4:11] -= arm_gravity(np.load(path)["q_mes"][keep])   # only for the kept frames
        if arrived:
            arrs.append(np.load(path)["nav_done"][keep].astype(np.float32))
        if vw_outputs:
            # Converted after picking the frames, so every output format trains on the same frames too
            y = np.concatenate([wheels_to_vw(y[:, :4]), y[:, 4:]], axis=1)
        xs.append(x)
        ys.append(y)
    if arrived:
        return np.concatenate(xs), np.concatenate(ys), np.concatenate(arrs)
    return np.concatenate(xs), np.concatenate(ys)


def load_frames(data_dir, episode_ids, keep_frac=None, mode="adaptive", seed=0, gravity_residual=False,
                all_labels=False):
    """For models that cut their windows per batch (TCN): every frame stored once, plus which ones are samples.

    Returns x (all frames of all episodes), y (labels of the kept frames only), frames (their rows in x) and starts
    (the row where each one's episode begins, so windows stop there). Same frames as load_episodes.
    all_labels: y has every frame's labels, in the same rows as x (the GRU scores whole windows).
    """
    xs, ys, frames, starts = [], [], [], []
    total = 0
    for i in episode_ids:
        path = Path(data_dir) / f"ep_{i:04d}.npz"
        x, y = load_episode(path)
        keep = np.arange(len(x))
        if keep_frac is not None:
            keep = subsample_indices(*decode_outputs(y), keep_frac=keep_frac, mode=mode, seed=(seed, i))
        labelled = np.arange(len(x)) if all_labels else keep
        y = y[labelled]
        if gravity_residual:
            y[:, 4:11] -= arm_gravity(np.load(path)["q_mes"][labelled])
        xs.append(x)
        ys.append(y)
        frames.append(keep + total)
        starts.append(np.full(len(keep), total))
        total += len(x)
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(frames), np.concatenate(starts)


def window_batch(x, frames, starts, n_frames, spacing):
    """Windows of the given rows of x: (len(frames), n_frames, n_in), oldest frame first, newest = the frame itself.

    Before an episode's first frame, that frame is repeated, as in with_history.
    """
    offsets = spacing * np.arange(n_frames - 1, -1, -1)
    return x[np.maximum(frames[:, None] - offsets, starts[:, None])]


def split_episodes(data_dir, seed=0, val_frac=0.15, test_frac=0.15):
    """Randomly assign whole episodes to train / validation / test (fixed seed, so always the same split)."""
    ids = sorted(int(p.stem[3:]) for p in Path(data_dir).glob("ep_*.npz"))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_test = round(len(ids) * test_frac)
    n_val = round(len(ids) * val_frac)
    test_ids = sorted(ids[:n_test])
    val_ids = sorted(ids[n_test:n_test + n_val])
    train_ids = sorted(ids[n_test + n_val:])
    return train_ids, val_ids, test_ids


def quick_test_ids(test_ids, n=50, seed=0):
    """Fixed subset of the test episodes for quick checks while developing (always the same ones for the same split).

    Drawn from the current test split: if the dataset changes (e.g. extended), the split and this subset change too.
    """
    rng = np.random.default_rng(seed)
    return sorted(int(i) for i in rng.choice(test_ids, size=min(n, len(test_ids)), replace=False))


def quick_val_ids(val_ids, n=50, seed=0):
    """Fixed subset of the validation episodes for closed-loop checks that decide something: compare.py --val-set,
    tune.py and dagger.py (decisions never use test episodes). 50 like the quick test subset, so differences
    between models aren't lost in the noise (tune.py used 20 until 30 Oct)."""
    return quick_test_ids(val_ids, n, seed)


REACH_TOLERANCE = 0.01   # m, loose check of the final error: the expert itself stops at 5 mm, and runs that
                         # drift to 5-10 mm in the 1 s settle after reaching are still good demonstrations
TORQUE_LIMIT = 30.0      # N*m, the expert clips the arm torques to +-this
FRAME_KEYS = ["base_pos", "base_bearing", "q_mes", "qd_mes", "wheel_cmds", "torque_arm", "ee_pos"]


def check_episode(d, min_duration=1.0):
    """Problems that make a saved episode unusable for training (empty list = clean). d is the loaded .npz."""
    problems = []
    message = str(d["stop_message"])
    if not message.startswith("Target reached"):
        problems.append(f"not reached ({message[:40]})")
    if float(d["final_error"]) > REACH_TOLERANCE:
        problems.append(f"final error {1000 * float(d['final_error']):.1f} mm")
    n = len(d["time"])
    if any(len(d[k]) != n for k in FRAME_KEYS):
        problems.append("arrays of different lengths")
    if not all(np.isfinite(d[k]).all() for k in FRAME_KEYS + ["time", "target"]):
        problems.append("NaN or inf values")
    if n == 0 or d["time"][-1] < min_duration:
        problems.append(f"shorter than {min_duration} s")
    return problems


def clean_ids(data_dir, episode_ids, max_saturated=0.1):
    """The episodes that pass check_episode. Prints what was dropped and why.

    Torques at the limit are expected (the expert's short kick when the reach starts), so they only give a warning
    when more than max_saturated of an episode's frames are saturated. Call this after split_episodes, on each set:
    dropping a bad episode then never moves a good one to another set.
    """
    kept, dropped, saturated = [], [], []
    for i in episode_ids:
        try:
            d = np.load(Path(data_dir) / f"ep_{i:04d}.npz")
            problems = check_episode(d)
            share = (np.abs(d["torque_arm"]) >= TORQUE_LIMIT - 0.01).any(axis=1).mean()
        except Exception as e:   # missing or truncated file (e.g. a crashed run)
            problems, share = [f"unreadable ({type(e).__name__})"], 0.0
        if problems:
            dropped.append((i, problems))
        else:
            kept.append(i)
            if share > max_saturated:
                saturated.append((i, share))
    print(f"episode check ({Path(data_dir).name}): kept {len(kept)} of {len(episode_ids)}")
    for i, problems in dropped:
        print(f"  dropped ep {i:04d}: {', '.join(problems)}")
    for i, share in saturated:
        print(f"  warning ep {i:04d}: torques at the limit in {100 * share:.1f}% of frames")
    return kept


if __name__ == "__main__":
    # Quick check: split sizes and the shapes of a few training episodes
    train_ids, val_ids, test_ids = split_episodes("data_main")
    print(len(train_ids), len(val_ids), len(test_ids))
    x, y = load_episodes("data_main", train_ids[:3])
    print(x.shape, y.shape, x.dtype)
