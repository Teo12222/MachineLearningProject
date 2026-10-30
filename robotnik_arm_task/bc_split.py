"""Split networks (PLAN.md step 15): a base network (wheels + "arrived" flag) and an arm network (torques).

Mirrors the expert's structure: the base drives until it has arrived, then stops, and the arm reaches. Each network
is a plain bc_model.MLP; both are saved together in one checkpoint, so the policy script and compare.py load a
split model by name like any other (load_policy).
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from bc_data import BASE_INPUTS, ARM_INPUTS
from bc_model import MLP


def base_inputs(x):
    """Base network input from make_features' 27 inputs (one frame or many)."""
    return x[..., BASE_INPUTS]


def arm_inputs(x, arrived):
    """Arm network input: its columns of make_features + the "arrived" flag (0/1) as the last input."""
    arrived = np.broadcast_to(np.asarray(arrived, dtype=np.float32), x.shape[:-1])[..., None]
    return np.concatenate([x[..., ARM_INPUTS], arrived], axis=-1)


class SplitPolicy(nn.Module):
    """Base network: 3 inputs -> 4 wheel commands + 1 "arrived" logit. Arm network: 25 inputs -> 7 torques."""

    def __init__(self, hidden=256, layers=3, dropout=0.0, gravity_residual=False):
        super().__init__()
        # kind tells load_policy which class to build; gravity_residual is read by whoever runs the model, as for the MLP
        self.config = dict(kind="split", hidden=hidden, layers=layers, dropout=dropout,
                           gravity_residual=gravity_residual)
        n_base = BASE_INPUTS.stop - BASE_INPUTS.start
        n_arm = ARM_INPUTS.stop - ARM_INPUTS.start + 1   # + the flag
        self.base = MLP(n_base, 5, hidden=hidden, layers=layers, dropout=dropout)
        self.arm = MLP(n_arm, 7, hidden=hidden, layers=layers, dropout=dropout)

    def set_normalisation(self, x, y, arrived):
        """Normalisation of both networks from the TRAINING frames (x: 27 inputs, y: 11 labels, arrived: 0/1)."""
        self.base.set_normalisation(base_inputs(x), base_labels(y, arrived))
        # The arrived output is a logit, not a quantity to rescale: mean 0 / std 1 leave it as it is
        self.base.y_mean[4], self.base.y_std[4] = 0.0, 1.0
        self.arm.set_normalisation(arm_inputs(x, arrived), y[:, 4:11])

    def predict_base(self, x):
        """Wheel commands (rad/s) and the probability that the base has arrived, from make_features' inputs."""
        out = self.base.predict(base_inputs(x))
        # Sigmoid written with tanh: the same curve, but no overflow warning for very negative logits
        return out[..., :4], 0.5 * (1.0 + np.tanh(0.5 * out[..., 4]))

    def predict_arm(self, x, arrived):
        """Arm torques (N*m; minus gravity for gravity-residual models) given the "arrived" flag (0/1)."""
        return self.arm.predict(arm_inputs(x, arrived))


def base_labels(y, arrived):
    """Base network labels: the 4 wheel commands + the arrived flag."""
    return np.concatenate([y[:, :4], arrived[:, None]], axis=1).astype(np.float32)


def base_loss(out, target, pos_weight):
    """MSE on the (normalised) wheel commands + weighted cross-entropy on the arrived logit, 1:1.

    pos_weight = arrived-False frames / arrived-True frames in the training data, so both classes count equally.
    """
    wheels = F.mse_loss(out[:, :4], target[:, :4])
    flag = F.binary_cross_entropy_with_logits(out[:, 4], target[:, 4], pos_weight=pos_weight)
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


def load_policy(path):
    """Load a checkpoint saved with bc_model.save_model: a SplitPolicy or a single MLP, whichever it holds."""
    ckpt = torch.load(path)
    config = dict(ckpt["config"])
    if config.pop("kind", None) == "split":
        model = SplitPolicy(**config)
    else:
        model = MLP(**config)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model
