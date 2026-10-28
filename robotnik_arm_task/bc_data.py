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


def load_episode(path):
    """Inputs and labels for every frame of one saved episode."""
    d = np.load(path)
    x = make_features(d["base_pos"], d["base_bearing"], d["q_mes"][:, 4:11], d["qd_mes"][:, 4:11], d["target"],
                      d["ee_pos"])
    y = make_labels(d["wheel_cmds"], d["torque_arm"])
    return x, y


def load_episodes(data_dir, episode_ids, every=1):
    """Inputs and labels of several episodes stacked into one array each.

    every: crude uniform subsample (every N-th frame), a placeholder until adaptive subsampling (PLAN.md step 7).
    """
    xs, ys = [], []
    for i in episode_ids:
        x, y = load_episode(Path(data_dir) / f"ep_{i:04d}.npz")
        xs.append(x[::every])
        ys.append(y[::every])
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


if __name__ == "__main__":
    # Quick check: split sizes and the shapes of a few training episodes
    train_ids, val_ids, test_ids = split_episodes("data_main")
    print(len(train_ids), len(val_ids), len(test_ids))
    x, y = load_episodes("data_main", train_ids[:3])
    print(x.shape, y.shape, x.dtype)
