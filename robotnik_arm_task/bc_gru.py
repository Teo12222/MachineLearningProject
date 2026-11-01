"""GRU training loop, shared by train_gru.py and the GRU halves of the split model (train.py --split, PLAN.md step 15).

A GRU learns from contiguous windows, with a burn-in and truncated backpropagation through time, and is scored on whole
episodes with its memory carried along, as at run time. The loss is passed in, so the same loop serves plain outputs
(MSE) and the base network (MSE + cross-entropy on the flag).
"""
import copy
import time

import numpy as np
import torch

from bc_data import window_batch

MAX_GRAD_NORM = 1.0      # long sequences can make the gradients explode


def predict_frames(net, x, frames, starts):
    """Output at the given rows of x (every frame of several episodes), real units, each episode run from its start."""
    begins = np.unique(starts)
    out = []
    for a, b in zip(begins, np.append(begins[1:], len(x))):
        out.append(net.predict(x[a:b][None])[0][frames[(frames >= a) & (frames < b)] - a])
    return np.concatenate(out)


def normalised(net, y):
    """Outputs in real units as a tensor in the network's normalised units."""
    return (torch.from_numpy(y) - net.y_mean) / net.y_std


def fit_gru(net, x, y, frames, starts, x_val, y_val, frames_val, starts_val, loss_fn, window=1000, batch=64, lr=1e-3,
            max_epochs=50, patience=10, label=""):
    """Train one GRU with early stopping; afterwards it holds the weights of its best validation epoch.

    x, y: every frame's inputs and labels in real units (load_frames with all_labels); frames, starts as load_frames,
    with frames the window ends. loss_fn(output, target) gets normalised rows (n, outputs) of the frames that count.
    Returns (best validation loss, best epoch, epochs run).
    """
    x = (x - net.x_mean.numpy()) / net.x_std.numpy()
    y = (y - net.y_mean.numpy()) / net.y_std.numpy()
    y_val_norm = normalised(net, y_val[frames_val])
    # An epoch shows about as many frames as the training set has, not one window per kept frame
    n_windows = len(x) // window
    n_frames = net.burn_in + window
    back = np.arange(n_frames - 1, -1, -1)
    print(f"{label}{n_windows} windows per epoch, {sum(p.numel() for p in net.parameters())} parameters")

    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    best_val, best_epoch, best_state = float("inf"), 0, None
    for epoch in range(1, max_epochs + 1):
        epoch_start = time.perf_counter()
        net.train()
        perm = torch.randperm(len(frames))[:n_windows].numpy()
        loss_sum = 0.0
        for i in range(0, len(perm), batch):
            ends, first = frames[perm[i:i + batch]], starts[perm[i:i + batch]]
            windows = window_batch(x, ends, first, n_frames, 1)
            labels = window_batch(y, ends, first, n_frames, 1)
            # Repeated first frames have no loss
            real = torch.from_numpy((ends[:, None] - back >= first[:, None])[:, net.burn_in:])
            out = net(torch.from_numpy(windows))[0][:, net.burn_in:]
            loss = loss_fn(out[real], torch.from_numpy(labels[:, net.burn_in:])[real])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), MAX_GRAD_NORM)
            opt.step()
            loss_sum += loss.item() * len(ends)

        net.eval()
        val_loss = loss_fn(normalised(net, predict_frames(net, x_val, frames_val, starts_val)), y_val_norm).item()
        print(f"{label}epoch {epoch:3d}  train {loss_sum / len(perm):.4f}  val {val_loss:.4f}  "
              f"({time.perf_counter() - epoch_start:.1f} s)", flush=True)
        if val_loss < best_val:
            best_val, best_epoch, best_state = val_loss, epoch, copy.deepcopy(net.state_dict())
        elif epoch - best_epoch >= patience:
            print(f"   no improvement for {patience} epochs: stopping (best epoch {best_epoch})")
            break
    net.load_state_dict(best_state)
    return best_val, best_epoch, epoch
