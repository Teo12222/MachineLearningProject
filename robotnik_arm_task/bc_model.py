"""Behaviour-cloning networks, with their normalisation stored alongside the weights."""
from collections import deque

import torch
from torch import nn


class MLP(nn.Module):
    """Baseline behaviour-cloning network. Stores its own normalisation so it can be used in real units."""

    def __init__(self, n_in, n_out, hidden=256, layers=3, dropout=0.0, gravity_residual=False,
                 history_k=0, history_spacing=1):
        super().__init__()
        # gravity_residual: the torque outputs are torque minus arm gravity (PLAN.md step 12). Not used by the
        # network itself; saved in the checkpoint so whoever runs the model knows to add gravity back
        # (decode_outputs). Older checkpoints don't have it and load with the default False.
        # history_k, history_spacing: past frames in each input (bc_data.with_history, PLAN.md step 14). Saved the same
        # way, so whoever runs the model builds the same inputs. n_in already counts them.
        self.config = dict(n_in=n_in, n_out=n_out, hidden=hidden, layers=layers, dropout=dropout,
                           gravity_residual=gravity_residual, history_k=history_k, history_spacing=history_spacing)
        dims = [n_in] + [hidden] * layers
        blocks = []
        for a, b in zip(dims[:-1], dims[1:]):
            blocks += [nn.Linear(a, b), nn.ReLU()]
            # Only added when used, so models without dropout keep the same layers (older checkpoints still load);
            # dropout is only active in training mode (model.train()), never in predict
            if dropout > 0:
                blocks.append(nn.Dropout(dropout))
        blocks.append(nn.Linear(dims[-1], n_out))
        self.net = nn.Sequential(*blocks)
        # Buffers are saved with the weights but are not trained
        self.register_buffer("x_mean", torch.zeros(n_in))
        self.register_buffer("x_std", torch.ones(n_in))
        self.register_buffer("y_mean", torch.zeros(n_out))
        self.register_buffer("y_std", torch.ones(n_out))

    def set_normalisation(self, x, y):
        """Mean/std of each input and output column, from the TRAINING data only (numpy arrays)."""
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        self.x_mean.copy_(x.mean(0))
        self.x_std.copy_(x.std(0).clamp(min=1e-3))   # clamp: near-constant columns (e.g. base z) would blow up
        self.y_mean.copy_(y.mean(0))
        self.y_std.copy_(y.std(0).clamp(min=1e-3))

    def forward(self, x_norm):
        """Normalised in, normalised out (used in training)."""
        return self.net(x_norm)

    @torch.no_grad()
    def predict(self, x):
        """Real units in, real units out (used in the simulation). x is a numpy array."""
        x = torch.as_tensor(x, dtype=torch.float32)
        y = self.net((x - self.x_mean) / self.x_std)
        return (y * self.y_std + self.y_mean).numpy()


class TCN(MLP):
    """Temporal convolutional network (PLAN.md step 14): causal, kernel 2, dilation doubling per layer.

    A layer combines its input now with its input spacing * 2^i ms earlier, so the network sees 2^layers frames,
    spacing ms apart. A kernel-2 convolution is a linear layer on the two time points joined, so it is written that
    way: forward (training) walks the whole tree, step (run time) computes only the newest node of each layer and keeps
    the rest in one ring buffer per layer. Normalisation and predict's units are the MLP's.
    """

    def __init__(self, n_in, n_out, layers=6, channels=64, spacing=10, gravity_residual=False):
        super().__init__(n_in, n_out, hidden=channels, layers=1, gravity_residual=gravity_residual)
        self.config = dict(arch="tcn", n_in=n_in, n_out=n_out, layers=layers, channels=channels, spacing=spacing,
                           gravity_residual=gravity_residual)
        dims = [n_in] + [channels] * layers
        self.convs = nn.ModuleList(nn.Linear(2 * a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.head = nn.Linear(channels, n_out)
        del self.net                              # the MLP's layers: only its normalisation is reused
        self.n_frames = 2 ** layers
        self.spacing = spacing
        self.reset()

    def forward(self, x_norm):
        """Normalised in, normalised out. x_norm: (batch, n_frames, n_in), oldest frame first."""
        h = x_norm
        for conv in self.convs:
            h = torch.relu(conv(torch.cat([h[:, 0::2], h[:, 1::2]], dim=-1)))   # each layer halves the sequence
        return self.head(h[:, 0])

    def reset(self):
        """Forget the past: call at the start of an episode."""
        self.buffers_by_layer = None

    @torch.no_grad()
    def step(self, x):
        """One simulation step: real units in, real units out. x: one frame of inputs (numpy array).

        Before the first frame there is no past: each buffer starts filled with its first value, which is what
        repeating the first frame gives (bc_data.window_batch).
        """
        h = (torch.as_tensor(x, dtype=torch.float32) - self.x_mean) / self.x_std
        if self.buffers_by_layer is None:
            self.buffers_by_layer = [deque(maxlen=self.spacing * 2 ** i) for i in range(len(self.convs))]
        for conv, buffer in zip(self.convs, self.buffers_by_layer):
            if not buffer:
                buffer.extend([h] * buffer.maxlen)
            past = buffer[0]                      # this layer's input spacing * 2^i ms ago
            buffer.append(h)
            h = torch.relu(conv(torch.cat([past, h], dim=-1)))
        return (self.head(h) * self.y_std + self.y_mean).numpy()

    @torch.no_grad()
    def predict(self, x):
        """Real units in, real units out. x: (batch, n_frames, n_in) numpy array, oldest frame first."""
        x = torch.as_tensor(x, dtype=torch.float32)
        return (self(((x - self.x_mean) / self.x_std)) * self.y_std + self.y_mean).numpy()


def save_model(model, path):
    torch.save({"config": model.config, "state_dict": model.state_dict()}, path)


def load_model(path):
    ckpt = torch.load(path)
    config = dict(ckpt["config"])
    model = (TCN if config.pop("arch", "mlp") == "tcn" else MLP)(**config)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model
