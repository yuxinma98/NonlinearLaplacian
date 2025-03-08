import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
import scipy.optimize as spo
from scipy.stats import norm


def double_gaussian_integral(f):
    """General calculation of E[f(x,y)] for x,y iid N(0, 1)."""
    return spi.dblquad(
        lambda x, y: f(x, y) * np.exp(-(x**2) / 2.0) * np.exp(-(y**2) / 2.0) / (2.0 * np.pi),
        -np.inf,
        np.inf,
        lambda x: -np.inf,
        lambda x: np.inf,
    )[0]


def H(z, sigma, c):
    """H function. H(z) = z + G_{\nu}(z). \nu = Law(\sigma(c\sqrt{2\pi} |g| + h))."""
    return z + double_gaussian_integral(
        lambda x, y: 1 / (z - sigma(c * np.sqrt(2 / np.pi) * np.abs(x) + y))
    )


def H_prime(z, sigma, c, sigma_image):
    """H function derivative."""
    if (
        z >= sigma_image[0] and z <= sigma_image[1]
    ):  # H_prime undefined when z is in the range of the support of nu
        return -np.inf
    return 1.0 - double_gaussian_integral(
        lambda x, y: 1 / (z - sigma(c * np.sqrt(2 / np.pi) * np.abs(x) + y)) ** 2
    )


def theta(c, sigma, sigma_image, tol):
    """Compute the effective signal."""

    def f(lam):  # lam_max is the root of this function, note it decreases in lam
        return (
            double_gaussian_integral(
                lambda g, d: g**2 / (lam - sigma(c * np.sqrt(2 / np.pi) * np.abs(g) + d))
            )
            - 1.0 / c
        )

    a = sigma_image[1] + 1e-7  # lower bound for lam_max
    b = c + sigma_image[1] + 1e-7  # upper bound for lam_max
    if f(a) * f(b) > 0:  # no solution for f(lam) = 0
        return sigma_image[1]  # so that H'(theta) is -inf
    lam_max = spo.bisect(f, a, b, xtol=tol)
    return lam_max


def c_critical(c_range, sigma, sigma_image, tol=2e-12, plot=True):
    """Critical value of c."""
    if plot:  # plot the diagram for theta(c) and H_prime(theta(c)) values
        c_values = np.linspace(c_range[0], c_range[1], 100)
        theta_values = np.array([theta(c, sigma, sigma_image, tol=tol) for c in c_values])
        H_prime_values = np.array(
            [
                H_prime(theta_values[i], sigma, c_values[i], sigma_image)
                for i in range(len(c_values))
            ]
        )
        # i = np.where(H_prime_values > -1)[0][0] # only plot the part where H_prime > -1
        i = 0
        fig, axes = plt.subplots(2, 1, sharex=True)
        axes[0].plot(c_values[i:], theta_values[i:])
        axes[1].plot(c_values[i:], H_prime_values[i:])
        axes[0].set_xlabel("$c$")
        axes[0].set_ylabel("$\\theta(c)$ (effective signal)")
        axes[1].set_xlabel("$c$")
        axes[1].set_ylabel("$H^{\\prime}(\\theta(c))$")
        plt.show()

    if (
        H_prime(theta(c_range[0], sigma, sigma_image, tol=tol), sigma, c_range[0], sigma_image)
        * H_prime(theta(c_range[1], sigma, sigma_image, tol=tol), sigma, c_range[1], sigma_image)
        > 0
    ):  # no solution for H'=0 in the range
        return c_range[1]
    c_critical = spo.bisect(
        lambda c: H_prime(theta(c, sigma, sigma_image, tol=tol), sigma, c, sigma_image),
        c_range[0],
        c_range[1],
        xtol=tol,
    )
    return c_critical


def outlier_eval(c, sigma, sigma_image, tol):
    """Evaluate the outlier eigenvalue when c > c_critical."""
    return H(theta(c, sigma, sigma_image, tol=tol), sigma, c)


# ########################## Discrete version for fast evaluation on step functions #########################


# def compute_cdfs(x, y, c):
#     """Compute the cdf of sigma(c\sqrt{2/pi} |g| + h), sigma is a step function"""
#     """input: x, y are np arrays define a step function f(x) = y[i] for x in (x[i], x[i+1])"""
#     cdf_values = np.zeros(len(x))
#     cdf_values[0] = 0
#     for i in range(1, len(x)):
#         cdf_values[i] = spi.quad(
#             lambda g: np.exp(-(g**2) / 2)
#             / np.sqrt(2 * np.pi)
#             * (
#                 norm.cdf(y[i] - c * np.sqrt(2 / np.pi) * np.abs(g))
#                 - norm.cdf(y[i - 1] - c * np.sqrt(2 / np.pi) * np.abs(g))
#             ),
#             -np.inf,
#             np.inf,
#         )[0]
#     return cdf_values.cumsum()


# def gaussian_integral(f):
#     """General calculation of E[f(x)] for x ~ N(0, 1)."""
#     return spi.quad(
#         lambda x: f(x) * np.exp(-(x**2) / 2.0) / np.sqrt(2.0 * np.pi), -np.inf, np.inf
#     )[0]


# def discrete_stieltjes(x, y, c):
#     """General calculation of E[f(x)] for x\sim sigma(c\sqrt{2/pi} |g| + h), where g,h\sim N(0,1)."""
#     """input: x, y are np arrays define a step function f(x) = y[i] for x in (x[i], x[i+1])"""
#     assert x[0] == -np.inf
#     assert x[-1] == np.inf
#     F_x = compute_cdfs(x, y, c)

#     F_x_diff = F_x[1:] - F_x[:-1]
#     return (F_x_diff * y).sum()


# def H_discrete(z, sigma, c):
#     """Subordination function."""
#     sigma_x, sigma_y = sigma
#     return z + discrete_stieltjes(sigma_x, 1.0 / (z - sigma_y), c)


# def H_prime_discrete(z, sigma, c, sigma_image):
#     """Subordination function derivative."""
#     if (
#         z >= sigma_image[0] and z <= sigma_image[1]
#     ):  # H_prime undefined when z is in the range of the support of nu
#         return -np.inf
#     sigma_x, sigma_y = sigma
#     return 1.0 - discrete_stieltjes(sigma_x, 1.0 / (z - sigma_y) ** 2, c)


# def c_critical_discrete(c_range, sigma, sigma_image, plot=True):
#     """Critical value of c."""
#     sigma_x, sigma_y = sigma
#     def sigma_fn(x):
#         idx = np.searchsorted(sigma_x, x) - 1
#         idx = np.clip(idx, 0, len(sigma_x) - 2)
#         return sigma_y[idx]

#     if plot:  # plot the diagram for theta(c) and H_prime(theta(c)) values
#         c_values = np.linspace(c_range[0], c_range[1], 100)
#         theta_values = np.array([theta(c, sigma_fn, sigma_image) for c in c_values])
#         H_prime_values = np.array(
#             [
#                 H_prime_discrete(theta_values[i], sigma, c_values[i], sigma_image)
#                 for i in range(len(c_values))
#             ]
#         )
#         # i = np.where(H_prime_values > -1)[0][0] # only plot the part where H_prime > -1
#         i = 0
#         fig, axes = plt.subplots(2, 1, sharex=True)
#         axes[0].plot(c_values[i:], theta_values[i:])
#         axes[1].plot(c_values[i:], H_prime_values[i:])
#         axes[0].set_xlabel("$c$")
#         axes[0].set_ylabel("$\\theta(c)$ (effective signal)")
#         axes[1].set_xlabel("$c$")
#         axes[1].set_ylabel("$H^{\\prime}(\\theta(c))$")
#         plt.show()

#     if (
#         H_prime_discrete(theta(c_range[0], sigma_fn, sigma_image), sigma, c_range[0], sigma_image)
#         * H_prime_discrete(theta(c_range[1], sigma_fn, sigma_image), sigma, c_range[1], sigma_image)
#         > 0
#     ):  # no solution for H'=0 in the range
#         return c_range[1]
#     c_critical = spo.bisect(
#         lambda c: H_prime_discrete(theta(c, sigma_fn, sigma_image), sigma, c, sigma_image),
#         c_range[0],
#         c_range[1],
#     )
#     return c_critical
