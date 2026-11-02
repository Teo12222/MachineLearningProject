"""Split networks (PLAN.md step 15): a base network (wheels + "arrived" flag) and an arm network (torques).

Mirrors the expert's structure: the base drives until it has arrived, then stops, and the arm reaches. Each network
is a bc_model.MLP or GRU; both are saved together in one checkpoint, so the policy script and compare.py load a
split model by name like any other (load_policy).
"""
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from bc_data import BASE_INPUTS, ARM_INPUTS, arm_gravity, make_features
from bc_model import GRU, MLP, load_model


def base_inputs(x):
    """Base network input from make_features' 27 inputs (one frame or many)."""
    return x[..., BASE_INPUTS]


def arm_inputs(x, arrived):
    """Arm network input: its columns of make_features + the "arrived" flag (0/1) as the last input."""
    arrived = np.broadcast_to(np.asarray(arrived, dtype=np.float32), x.shape[:-1])[..., None]
    return np.concatenate([x[..., ARM_INPUTS], arrived], axis=-1)


def net_settings(config, part):
    """(hidden, layers, dropout) of a split model's "base" or "arm" network, from its config: the network's own
    setting where one is given (base_hidden, arm_dropout, ...), else the shared one (hidden, layers, dropout). Split
    checkpoints from before the per-network settings have none, so both networks use the shared ones."""
    return tuple(config[k] if config.get(f"{part}_{k}") is None else config[f"{part}_{k}"]
                 for k in ("hidden", "layers", "dropout"))


class SplitPolicy(nn.Module):
    """Base network: 3 inputs -> 4 wheel commands (or v, w) + 1 "arrived" logit. Arm network: 25 inputs -> 7 torques.

    Labels and base outputs are indexed from the end (torques = last 7 labels, logit = last base output), so the same
    code serves both wheel formats. Either network can be a GRU (base_arch, arm_arch): a GRU steps every frame from
    t = 0, the arm with flag 0 while the base drives, so its memory is warm at the latch.
    """

    def __init__(self, hidden=256, layers=3, dropout=0.0, gravity_residual=False, vw_outputs=False, base_arch="mlp",
                 arm_arch="mlp", gru_hidden=192, burn_in=200, base_hidden=None, base_layers=None, base_dropout=None,
                 arm_hidden=None, arm_layers=None, arm_dropout=None):
        super().__init__()
        # kind tells load_policy which class to build; gravity_residual and vw_outputs are read by whoever runs the
        # model (decode_outputs), as for the MLP
        self.config = dict(kind="split", hidden=hidden, layers=layers, dropout=dropout,
                           gravity_residual=gravity_residual, vw_outputs=vw_outputs, base_arch=base_arch,
                           arm_arch=arm_arch, gru_hidden=gru_hidden, burn_in=burn_in,
                           # Per-network size and dropout (None: the shared one, see net_settings), so the arm can
                           # be tuned on a copied base of another size (PLAN.md 4a)
                           base_hidden=base_hidden, base_layers=base_layers, base_dropout=base_dropout,
                           arm_hidden=arm_hidden, arm_layers=arm_layers, arm_dropout=arm_dropout)
        n_base = BASE_INPUTS.stop - BASE_INPUTS.start
        n_arm = ARM_INPUTS.stop - ARM_INPUTS.start + 1   # + the flag
        n_wheel = 2 if vw_outputs else 4

        def network(arch, n_in, n_out, part):
            if arch == "gru":
                return GRU(n_in, n_out, hidden=gru_hidden, burn_in=burn_in)
            h, n, p = net_settings(self.config, part)
            return MLP(n_in, n_out, hidden=h, layers=n, dropout=p)

        self.base = network(base_arch, n_base, n_wheel + 1, "base")
        self.arm = network(arm_arch, n_arm, 7, "arm")

    def reset(self):
        """Forget the past: call at the start of an episode (GRU networks only have any)."""
        for net in (self.base, self.arm):
            if isinstance(net, GRU):
                net.reset()

    def set_normalisation(self, x, y, arrived):
        """Normalisation of both networks from the TRAINING frames (x: 27 inputs, y: wheel + 7 torque labels,
        arrived: 0/1)."""
        self.base.set_normalisation(base_inputs(x), base_labels(y, arrived))
        # The arrived output is a logit, not a quantity to rescale: mean 0 / std 1 leave it as it is
        self.base.y_mean[-1], self.base.y_std[-1] = 0.0, 1.0
        self.arm.set_normalisation(arm_inputs(x, arrived), y[:, -7:])

    def predict_base(self, x):
        """Wheel outputs (4 commands in rad/s, or v, w) and the probability that the base has arrived.

        x: a whole episode from its start, or for an MLP any frames.
        """
        return wheels_and_p(run_frames(self.base, base_inputs(x)))

    def predict_arm(self, x, arrived):
        """Arm torques (N*m; minus gravity for gravity-residual models) given the "arrived" flag (0/1).

        x: a whole episode from its start, or for an MLP any frames.
        """
        return run_frames(self.arm, arm_inputs(x, arrived))

    def step_base(self, x):
        """predict_base for one simulation step (x: one frame), keeping a GRU's memory."""
        return wheels_and_p(step_frame(self.base, base_inputs(x)))

    def step_arm(self, x, arrived):
        """predict_arm for one simulation step (x: one frame), keeping a GRU's memory."""
        return step_frame(self.arm, arm_inputs(x, arrived))


def run_frames(net, x):
    return net.predict(x[None])[0] if isinstance(net, GRU) else net.predict(x)


def step_frame(net, x):
    return net.step(x) if isinstance(net, GRU) else net.predict(x)


def wheels_and_p(out):
    # Sigmoid written with tanh: the same curve, but no overflow warning for very negative logits
    return out[..., :-1], 0.5 * (1.0 + np.tanh(0.5 * out[..., -1]))


def base_labels(y, arrived):
    """Base network labels: the wheel labels (4 commands, or v, w) + the arrived flag."""
    return np.concatenate([y[:, :-7], arrived[:, None]], axis=1).astype(np.float32)


def base_loss(out, target, pos_weight):
    """MSE on the (normalised) wheel outputs + weighted cross-entropy on the arrived logit, 1:1.

    pos_weight = arrived-False frames / arrived-True frames in the training data, so both classes count equally.
    """
    wheels = F.mse_loss(out[:, :-1], target[:, :-1])
    flag = F.binary_cross_entropy_with_logits(out[:, -1], target[:, -1], pos_weight=pos_weight)
    return wheels + flag


class ArrivedLatch:
    """Turns the base network's p(arrived) into the flag that switches control from the base to the arm.

    True once p > threshold for `hold` consecutive steps (1 ms each, so a single flicker can't switch), or once the
    time passes `timeout` (the expert's own navigation timeout: the arm then reaches from wherever the base is).
    Never False again, like the expert's flag. `source` records what latched it ("network" or "timeout").
    The same rule is used by the offline test (train.py) and the policy script.
    """

    def __init__(self, threshold=0.5, hold=20, timeout=25.0):
        self.threshold, self.hold, self.timeout = threshold, hold, timeout
        self.count, self.arrived, self.source = 0, False, None

    def update(self, p, t):
        if not self.arrived:
            self.count = self.count + 1 if p > self.threshold else 0
            if self.count >= self.hold:
                self.arrived, self.source = True, "network"
            elif t > self.timeout:   # same comparison as the expert's base_controller
                self.arrived, self.source = True, "timeout"
        return self.arrived


def load_dagger_arm(folders, step, gravity_residual):
    """Arm-network training frames from DAgger episodes (dagger.py output, PLAN.md step 21).

    Every step-th frame with an expert label (arm_label_mask: the arm phase after the latch), as arm-network inputs
    with the flag set, and the expert's relabelled torques as labels (minus gravity for gravity-residual models).
    The episodes store the full torque, so the same data serves models with and without the gravity residual.
    DART episodes (run_batch.py --noise, PLAN.md step 19) come from the expert, so every frame has a label: all count,
    with the expert's own flag.
    """
    xs, ys = [], []
    for path in sorted(p for folder in folders for p in Path(folder).glob("ep_*.npz")):
        d = np.load(path)
        dagger = "arm_label_mask" in d.files
        keep = np.flatnonzero(d["arm_label_mask"] if dagger else np.ones(len(d["time"]), dtype=bool))[::step]
        x = make_features(d["base_pos"][keep], d["base_bearing"][keep], d["q_mes"][keep, 4:11], d["qd_mes"][keep, 4:11],
                          d["target"], d["ee_pos"][keep])
        y = d["torque_arm"][keep].astype(np.float32)
        if gravity_residual:
            y -= arm_gravity(d["q_mes"][keep])
        xs.append(arm_inputs(x, 1.0 if dagger else d["nav_done"][keep]))
        ys.append(y)
    return np.concatenate(xs), np.concatenate(ys)


def load_policy(path):
    """Load a checkpoint saved with bc_model.save_model: a SplitPolicy, or whatever single network it holds."""
    ckpt = torch.load(path)
    config = dict(ckpt["config"])
    if config.pop("kind", None) != "split":
        return load_model(path)   # MLP, TCN or GRU (PLAN.md step 14): bc_model picks the class
    model = SplitPolicy(**config)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model
