"""Data side of behaviour cloning: model inputs/outputs, loading saved episodes, and the episode split.

make_features is shared by training (whole episodes) and robotnik_arm_policy_task.py (one frame per step),
so the model always sees inputs built the same way.
"""
import numpy as np
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


def make_labels(wheel_cmds, torque_arm):
    """Model output: 4 wheel commands then 7 arm torques."""
    return np.concatenate([wheel_cmds, torque_arm], axis=-1).astype(np.float32)


def decode_outputs(y):
    """Model output back to robot commands: (4 wheel commands in rad/s, 7 arm torques in N*m).

    The inverse of make_labels. Works for one frame or a whole episode (one row per frame).
    Everything that runs or scores a model (policy script, offline test) goes through this, so a new output
    format (e.g. (v, w) instead of 4 wheel speeds) only changes make_labels and this function.
    """
    y = np.asarray(y)
    return y[..., :4], y[..., 4:11]


def load_episode(path):
    """Inputs and labels for every frame of one saved episode."""
    d = np.load(path)
    x = make_features(d["base_pos"], d["base_bearing"], d["q_mes"][:, 4:11], d["qd_mes"][:, 4:11], d["target"],
                      d["ee_pos"])
    y = make_labels(d["wheel_cmds"], d["torque_arm"])
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


def load_episodes(data_dir, episode_ids, keep_frac=None, mode="adaptive", seed=0):
    """Inputs and labels of several episodes stacked into one array each.

    keep_frac=None keeps every frame; otherwise each episode keeps only the frames picked by subsample_indices.
    """
    xs, ys = [], []
    for i in episode_ids:
        x, y = load_episode(Path(data_dir) / f"ep_{i:04d}.npz")
        if keep_frac is not None:
            keep = subsample_indices(*decode_outputs(y), keep_frac=keep_frac, mode=mode, seed=(seed, i))
            x, y = x[keep], y[keep]
        xs.append(x)
        ys.append(y)
    return np.concatenate(xs), np.concatenate(ys)


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


def quick_val_ids(val_ids, n=20, seed=0):
    """Fixed subset of the validation episodes for tune.py's closed-loop check (tuning never uses test episodes)."""
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
