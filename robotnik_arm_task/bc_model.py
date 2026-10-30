"""Baseline behaviour-cloning network, with its normalisation stored alongside the weights."""
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


def save_model(model, path):
    torch.save({"config": model.config, "state_dict": model.state_dict()}, path)


def load_model(path):
    ckpt = torch.load(path)
    model = MLP(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model
