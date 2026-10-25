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


def load_episode(path):
    """Inputs and labels for every frame of one saved episode."""
    d = np.load(path)
    x = make_features(d["base_pos"], d["base_bearing"], d["q_mes"][:, 4:11], d["qd_mes"][:, 4:11], d["target"])
    y = make_labels(d["wheel_cmds"], d["torque_arm"])
    return x, y


def load_episodes(data_dir, episode_ids):
    """Inputs and labels of several episodes stacked into one array each."""
    xs, ys = [], []
    for i in episode_ids:
        x, y = load_episode(Path(data_dir) / f"ep_{i:04d}.npz")
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
