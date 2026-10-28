"""Data side of behaviour cloning: model inputs/outputs, loading saved episodes, and the episode split.

make_features is shared by training (whole episodes) and robotnik_arm_policy_task.py (one frame per step),
so the model always sees inputs built the same way.
"""
import numpy as np
from pathlib import Path


def make_features(base_pos, base_bearing, q_arm, qd_arm, target):
    """Model input. Works for one frame (1-D arrays) or a whole episode (one row per frame)."""
    base_bearing = np.asarray(base_bearing)
    return np.concatenate([
        base_pos,                                  # base x, y, z            (3)
        base_bearing[..., None],                   # heading in radians      (1)
        np.sin(base_bearing)[..., None],           # heading as sin/cos, no jump at +-pi (2)
        np.cos(base_bearing)[..., None],
        q_arm,                                     # arm joint angles        (7)
        qd_arm,                                    # arm joint velocities    (7)
        np.broadcast_to(target, base_pos.shape),   # target, copied to every frame (3)
    ], axis=-1).astype(np.float32)


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
    x = make_features(d["base_pos"], d["base_bearing"], d["q_mes"][:, 4:11], d["qd_mes"][:, 4:11], d["target"])
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


if __name__ == "__main__":
    # Quick check: split sizes and the shapes of a few training episodes
    train_ids, val_ids, test_ids = split_episodes("data_pilot")
    print(len(train_ids), len(val_ids), len(test_ids))
    x, y = load_episodes("data_pilot", train_ids[:3])
    print(x.shape, y.shape, x.dtype)
