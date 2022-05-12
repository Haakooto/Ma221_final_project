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
    """
    Fully connected neural network.
    """
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
        # string togheter linear layers with non-linearities inbetween
        self.layers = nn.Sequential(*layers,
                                    nn.Linear(
                                        nodes[-2], nodes[-1], bias=use_bias, dtype=torch.float),
                                    )
        self.to(torch.float)

        # items to record
        self.losses = []
        self.vals = []
        self.vecs = []
        self.residuals = []

        self.epochs = 0

    def forward(self, x):  # pass input through every layer
        return self.layers(x)

    def train(self, S_, t, epochs, lr=0.001):
        optimer = torch.optim.AdamW(self.parameters(), lr=lr)

        # convert matrices to torch tensors
        t = torch.tensor([t]).t().to(torch.float)
        t.requires_grad = True
        S = torch.tensor(S_).to(torch.float)
        change = 0  # initialize

        pbar = tqdm(range(epochs))  # progress bar
        try:  # catch keyboard interuptions
            for i in pbar:
                optimer.zero_grad()

                x = self(t)  # pass input through network
                jac = torch.autograd.functional.jacobian(
                    self, t, create_graph=True)  # differentiate output w.r.t input
                dt = einops.reduce(jac, "d n t s -> d n", reduction="sum").t()   # dx/dt

                Ax = x @ S  # the reverse order lets many time-points t to be passed through and calculate loss correctly
                xxAx = (x * x).sum(1) * Ax.t()
                xAxx = (x * Ax).sum(1) * x.t()

                loss = einops.reduce((dt - xxAx + xAxx)**2,
                                    "n t -> t", reduction="mean").sum()
                loss.backward()
                optimer.step()

                # record several statistics
                evec = x[-1].detach().numpy()  # vector resulting from highest time t
                evec *= np.sign(evec[0])  # make the first element positive
                normed_vec = evec / np.linalg.norm(evec)
                eval = normed_vec.T @ S_ @ normed_vec
                res = S_ @ normed_vec - eval * normed_vec

                self.residuals.append(amax(res))
                self.vecs.append(normed_vec)
                self.losses.append(loss.detach())
                self.vals.append(eval)

                if i > 10:
                    change = np.abs(np.mean(self.vals[-10:-1]) - self.vals[-1])
                    if change < 1e-6:
                        print("Network converged!")
                        break

                pbar.set_description(f"eval: {eval:.5f},\
                                  log(loss): {torch.log10(loss.data):.5f},\
                                   log(res): {np.log10(amax(res)):.5f},\
                                       ")

        except KeyboardInterrupt:
            print("Interrupted!")

        self.epochs += i

        self.eval = self.vals[-1]
        self.evec = self.vecs[-1]


def eigsort(A):
    """
    Useful function for sorting eigenvalues and vectors
    from np.linalg.eig by the magnitude of eigenvalue
    Also ensures that the first element in the eigenvectors is positive
    """
    vals, vecs = np.linalg.eig(A)
    idxs = np.argsort(vals)
    vals = vals[idxs]
    vecs = vecs[:, idxs]
    vecs *= np.sign(vecs[0, :])
    return vals, vecs


def plot_history(model, evals, evecs, name):

    fig, axs = plt.subplots(2, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [4, 2]})
    ax_val, ax_err = axs

    ax_val.plot(model.vals, "orange", lw=4)
    ax_val.plot(einops.repeat(evals, "v -> r v", r=model.epochs), "r--", lw=2)
    ax_val.set_title(f"Calculated eigenvalue. Final value: {model.vals[-1]:.10f}")
    ax_val.legend(["Potential eigenvalue", "true eigenvalues"])
    ax_val.set_ylabel("value of eigenvalue")

    ax_err.plot(np.log10([np.min(np.abs(model.vals[i] - evals)) for i in range(model.epochs)]), "k", label="distance to nearest eigenvalue")
    ax_err.plot(np.log10(model.residuals), "b", label=r"residual, Ax'-$\lambda$'x'")
    ax_err.set_title(f"Number of accurate decimals: {-np.log10(np.min(np.abs(evals - model.vals[-1]))):.0f}")
    ax_err.set_ylabel("log10(y)")
    ax_err.set_xlabel("epochs")
    ax_err.legend()

    plt.savefig(f"figs/{name}.pdf")
    plt.clf()


def plot_vecs(model, name):
    n = model.vecs[0].shape[0]
    fig, axs = plt.subplots(figsize=(12, 8))
    cols = mpl.cm.brg(np.linspace(0, 1, n))
    mvecs = np.column_stack(model.vecs)
    for c in range(n):
        axs.plot(mvecs[c, :], color=cols[c])
        axs.legend([f"v{i}" for i in range(n)])
        axs.set_xlabel("epochs")
        axs.set_ylabel("value")
        axs.set_title(f"Output of network during training")
    plt.savefig(f"figs/{name}.pdf")
    plt.clf()
