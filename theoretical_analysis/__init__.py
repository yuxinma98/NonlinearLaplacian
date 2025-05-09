import numpy as np
import torch
import scipy.integrate as spi
import matplotlib.pyplot as plt


def gaussian_integral(f):
    """General calculation of E[f(x)] for x ~ N(0, 1)."""
    return spi.quad(
        lambda x: f(x) * np.exp(-(x**2) / 2.0) / np.sqrt(2.0 * np.pi), -np.inf, np.inf
    )[0]


def beta_to_sigma(beta):
    """
    Given beta (parameters for step function that's convenient for optimization), output x,y (defines the step function)
    beta = (a[0], a[1], ..., a[n], b[0], ..., b[n])
    Define sigma to be a step function parametrized by:
    f(x) = y[i] for x in (x[i], x[i+1]) where
    x = (-np.inf, a[0], a[0:1].sum(), ..., a[0:n].sum(), np.inf)
    y = (0, b[0], b[0:1].sum(), ..., b[0:n].sum(), b[0:n].sum())
    """
    n = len(beta) // 2 - 1
    a, b = beta[0 : n + 1], beta[n + 1 :]
    x = np.cumsum(a)
    x = np.concatenate(([-np.inf], x, [np.inf]))
    y = np.cumsum(b)
    y = np.concatenate(([0], y))
    return x, y


def step_function(sigma, t):
    """Given parametrization (x,y) of the step function, evaluate the function at t."""
    x, y = sigma
    n = len(x) - 3
    if isinstance(t, int) or isinstance(t, float):
        for i in range(0, n + 2):
            if t >= x[i] and t < x[i + 1]:
                return y[i]
    elif isinstance(t, np.ndarray):
        y_t = np.zeros_like(t)
        for i in range(0, n + 2):
            y_t[(t >= x[i]) & (t < x[i + 1])] = y[i]
        return y_t
    elif isinstance(t, torch.Tensor):
        y_t = torch.zeros_like(t)
        for i in range(0, n + 2):
            y_t[(t >= x[i]) & (t < x[i + 1])] = y[i]
        return y_t


def plot_step_function(
    beta,
    fname=None,
    figsize=(4, 3),
    axes_rect=[0.2, 0.1, 0.7, 0.8],
    fixed_y_range=False,
    fontsize=18,
    offset=0.05,
):
    """Plot the step function given the parametrization beta.

    Args:
        beta (np.array): parametrization of the step function
        fname (str, optional): If not none, save the plot with this fname. Defaults to None.
    """
    x, y = beta_to_sigma(beta)
    x_s = np.linspace(-10, 10, 1000)
    y_s = step_function((x, y), x_s)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(axes_rect)
    ax.plot(x_s, y_s, color="black", linewidth=3)
    ax.set_xticks([x[1], x[-2]])
    ax.set_xticklabels([f"{x[1]:.1f}", f"{x[-2]:.1f}"], fontsize=fontsize)
    ax.set_yticks([0, y[-1]])
    ax.set_yticklabels([0, f"{y[-1]:.1f}"], fontsize=fontsize)
    ax.vlines(x[1], y[0] - 10, y[-1] + 10, linestyle="--", color="gray", zorder=1)
    ax.vlines(x[-2], y[0] - 10, y[-1] + 10, linestyle="--", color="gray", zorder=1)
    ax.hlines(y[-1], -10, 10, linestyle="--", color="gray", zorder=1)
    ax.hlines(y[0], -10, 10, linestyle="--", color="gray", zorder=1)
    ax.set_xlim(-10, 10)
    if fixed_y_range:
        ax.set_ylim(y[0] - offset * 5.3, y[0] + 5.3 + offset * 5.3)
        fname = fname.replace(".pdf", "_fixed_y.pdf")
    else:
        range = y[-1] - y[0]
        ax.set_ylim(y[0] - offset * range, y[-1] + offset * range)
    plt.show()
    if fname:
        fig.savefig(fname)
    plt.close(fig)
