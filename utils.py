import torch
from torch import nn
from torch.nn import functional as F
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import einops
from tqdm.notebook import tqdm

amax = lambda x: np.max(np.abs(x))


class EigenSolver(nn.Module):
    def __init__(self, n, hiddens, nonlin=nn.ReLU, use_bias=True, seed=None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)

        self.activation = nonlin
        nodes = [1] + [int(h) for h in hiddens] + [n]

        layers = []
        for prev, next in zip(nodes[:-2], nodes[1:-1]):
            layers += [nn.Linear(prev, next, bias=use_bias,
                                 dtype=torch.float), self.activation()]

        self.layers = nn.Sequential(*layers,
                                    nn.Linear(
                                        nodes[-2], nodes[-1], bias=use_bias, dtype=torch.float),
                                    )
        self.to(torch.float)

        # items to record
        self.losses = []
        self.vals = []
        self.jacsum = []
        self.vecs = []
        self.residuals = []
        self.maxdts = []

        self.epochs = 0

    def forward(self, x):
        return self.layers(x)

    def train(self, S_, t, epochs, lr=0.001):
        optimer = torch.optim.AdamW(self.parameters(), lr=lr)

        t = torch.tensor([t]).t().to(torch.float)
        t.requires_grad = True
        S = torch.tensor(S_).to(torch.float)
        S.requires_grad = True
        change = 0

        pbar = tqdm(range(epochs))
        try:
            for i in pbar:
                optimer.zero_grad()

                x = self(t)  # pass input through network
                jac = torch.autograd.functional.jacobian(
                    self, t, create_graph=True)
                dt = einops.reduce(jac, "d n t s -> d n", reduction="sum").t()   # dx/dt

                Ax = x @ S
                xxAx = (x * x).sum(1) * Ax.t()
                xAxx = (x * Ax).sum(1) * x.t()

                loss = einops.reduce((dt - xxAx + xAxx)**2,
                                    "n t -> t", reduction="mean").sum()
                loss.backward()
                optimer.step()

                # record several statistics
                evec = x[-1].detach().numpy()  # vector resulting from highest time t
                normed_vec = evec / np.linalg.norm(evec)
                eval = normed_vec.T @ S_ @ normed_vec
                res = S_ @ normed_vec - eval * normed_vec

                self.residuals.append(amax(res))
                self.vecs.append(normed_vec)
                self.losses.append(loss.detach())
                self.vals.append(eval)
                self.jacsum.append(jac.detach().abs().sum())
                self.maxdts.append(dt.detach().abs().max())

                if i > 5:
                    change = np.max(np.abs((normed_vec - self.vecs[-2])))
                    if change < 1e-6:
                        print("Vector stopped moving!")
                        break
                if np.max(np.abs(res)) < 1e-6:
                    print(f"Reached tolerance residual. Vector is {'NOT' if (normed_vec < 1e-6).all() else ''} the 0-vector!")
                    break

                pbar.set_description(f"eval: {eval:.5f},\
                                  log(loss): {torch.log10(loss.data):.5f},\
                                   log(res): {np.log10(amax(res)):.5f},\
                                     change: {change:10f},\
                                       ")

        except KeyboardInterrupt:
            print("Interrupted!")

        self.epochs += i

        self.eval = self.vals[-1]
        self.evec = self.vecs[-1]


def plot_history(model, evals, evecs):
    fig, axs = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [4, 2]})
    ax_val, ax_dt = axs
    ax_loss = ax_dt.twinx()
    ax_err = ax_val.twinx()

    ax_val.plot(model.vals, "orange", lw=4)

    ax_err.plot(np.log10([np.min(np.abs(model.vals[i] - evals)) for i in range(model.epochs)]), "k", label="absolute difference to nearest true eval")
    ax_err.plot(np.log10(model.residuals), "b", label="residual")
    ax_err.set_ylabel("log(error)")
    ax_err.legend(loc=1)
    ax_val.plot(einops.repeat(evals, "v -> r v", r=model.epochs), "r--", lw=2)
    ax_val.legend(["eigenvalue during training", "true eigenvalues"], loc=9)
    ax_val.set_ylabel("value of eigenvalue")
    ax_val.set_xlabel("epochs")

    ax_dt.plot(np.log10(np.asarray(model.maxdts) + 1e-16), "b", label="max(dt)")
    ax_dt.set_ylabel("log10(dt)")
    ax_loss.plot(np.log10(model.losses), "k", label="log10(loss)")
    ax_loss.set_ylabel("log10(loss)")
    ax_dt.legend()

    ax_val.set_title(f"value at end: {model.vals[-1]:.5f}, log of abs error from closes true val: {np.log10(np.min(np.abs(evals - model.vals[-1]))):.5f}")
    ax_dt.set_title("max(dt) during training")


def eigsort(A):
    vals, vecs = np.linalg.eig(A)
    idxs = np.argsort(vals)
    vals = vals[idxs]
    vecs = vecs[:, idxs]
    return vals, vecs