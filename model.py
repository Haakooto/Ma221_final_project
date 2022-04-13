import torch
from torch import nn
import torch.nn.functional as F
from tqdm.notebook import tqdm
import numpy as np


class FullyConnectedN(nn.Module):
    """
    Fully connected model with N layes
    """
    nonlin = {"relu": nn.ReLU, "softplus": nn.Softplus}

    def __init__(self, n, hidden_nodes=[10, 10], nonlinearity="relu"):
        super().__init__()

        # choose non-linearity between layers
        self.act = FullyConnectedN.nonlin[nonlinearity]
        nodes = [1] + hidden_nodes + [n]

        # String together layers in a list
        layers = []
        for prev, next in zip(nodes[:-2], nodes[1:-1]):
            layers += [nn.Linear(prev, next), self.act()]

        # Unpack list into a nn.Sequential, and add a last layer without non-linearity
        self.layers = nn.Sequential(
            *layers, nn.Linear(nodes[-2], nodes[-1]))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def train(model, optimizer, A, x0, t, epochs=1, device=None, dtype=None):
    model = model.to(device=device)
    model.train()

    xx0 = torch.tensor(x0.T @ x0).to(device=device, dtype=dtype)
    t = torch.tensor(t).reshape(-1, 1).to(device=device, dtype=dtype)
    t.requires_grad = True
    A = torch.tensor(A).to(device=device, dtype=dtype)
    n = A.size(1)

    pbar = tqdm(range(epochs))
    for e in pbar:
        optimizer.zero_grad()

        out = model(t)
        dx = torch.zeros_like(out)
        for i in range(n):
            dx[:, i] = torch.autograd.grad(
                outputs=out[0][i], inputs=t, grad_outputs=torch.ones_like(out[0][i]), retain_graph=True)[0]
        x = out.reshape(-1, 1)
        Ax = A @ x
        xxAx = (x.T @ x) * Ax
        # xxAx = xx0 * Ax
        xAxx = (x.T @ Ax) * x

        lmb = (x.T @ A @ x) / (x.T @ x)
        lmb = lmb[0][0]

        # d = dx.T - xxAx + xAxx
        # d = d / torch.norm(d)

        diff = torch.mean((dx.T - xxAx + xAxx)**2)

        # loss = F.mse_loss(dx.T, xxAx - xAxx)
        # loss = F.mse_loss(d, torch.zeros_like(d))

        diff.backward()
        optimizer.step()
        # scheduler.step()

        pbar.set_description(f"Loss: {diff:.7f}, lmb: {lmb.data:.4f}")
    return x.detach().numpy()


def compare_eig(S, eigv):
    scaled = eigv / np.linalg.norm(eigv)
    lmb = (eigv.T @ S @ eigv) / (eigv.T @ eigv)
    lmb = lmb[0]

    # get true eig_vecs using numpy.linalg.eig
    true_vals, true_vecs = np.linalg.eig(S)

    # ensure the same sign of the eigenvectors
    true_vecs *= np.sign(true_vecs[0])
    scaled *= np.sign(scaled[0])

    idx = np.argmin(abs(true_vals - lmb))
    true_vec = true_vecs[:, idx].reshape(-1, 1)
    true_val = true_vals[idx]

    val_rel_error = abs(lmb - true_val)
    vec_rel_error = max(abs(scaled - true_vec))

    return {"evals_np": true_vals, "eval_np": true_val, "eval_NN": lmb[0], "val_err": val_rel_error[0], "evecs_np": true_vecs, "evec_np": true_vec, "evec_NN": scaled, "vec_err": vec_rel_error[0]}
