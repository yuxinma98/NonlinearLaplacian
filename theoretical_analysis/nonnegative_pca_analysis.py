import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
import scipy.optimize as spo
from scipy.stats import norm
from theoretical_analysis import gaussian_integral, beta_to_sigma


def double_gaussian_integral(f):
    """General calculation of E[f(x,y)] for x,y iid N(0, 1)."""
    return spi.nquad(
        lambda x, y: f(x, y) * np.exp(-(x**2) / 2.0) * np.exp(-(y**2) / 2.0) / (2.0 * np.pi),
        [[-np.inf, np.inf], [-np.inf, np.inf]],
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

    # def f_prime(lam):
    #     return -double_gaussian_integral(
    #         lambda g, d: g**2 / (lam - sigma(c * np.sqrt(2 / np.pi) * np.abs(g) + d)) ** 2
    #     )

    a = sigma_image[1] + 1e-7  # lower bound for lam_max
    b = c + sigma_image[1] + 1e-7  # upper bound for lam_max
    if f(a) * f(b) > 0:  # no solution for f(lam) = 0
        return sigma_image[1]  # so that H'(theta) is -inf
    # lam_max = spo.newton(f, a, fprime=f_prime, tol=tol)
    lam_max = spo.brentq(f, a, b, xtol=tol)
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
    c_critical = spo.brentq(
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


def compute_cdfs(x, c):
    """Compute the cdf of c\sqrt{2/pi} |g| + h"""
    sqrt_2_pi = np.sqrt(2 / np.pi)

    return np.vectorize(
        lambda t: gaussian_integral(lambda g: norm.cdf(t - c * sqrt_2_pi * np.abs(g)))
    )(x)


def discrete_stieltjes(x, y, c):
    """General calculation of E[f(x)] for step function f, where x\sim sigma(c\sqrt{2/pi} |g| + h), where g,h\sim N(0,1)."""
    """input: x, y are np arrays define a step function f(x) = y[i] for x in (x[i], x[i+1])"""
    assert x[0] == -np.inf
    assert x[-1] == np.inf
    F_x = compute_cdfs(x, c)

    F_x_diff = F_x[1:] - F_x[:-1]
    return (F_x_diff * y).sum()


def H_discrete(z, sigma, c):
    """Subordination function. H(z) = z + G_{\nu}(z). \nu = Law(\sigma(c\sqrt{2\pi} |g| + h))."""
    sigma_x, sigma_y = sigma
    return z + discrete_stieltjes(sigma_x, 1.0 / (z - sigma_y), c)


def H_prime_discrete(z, sigma, c):
    """Subordination function derivative."""
    if (
        z >= 0 and z <= sigma[1][-1]
    ):  # H_prime undefined when z is in the range of the support of nu
        return -np.inf
    sigma_x, sigma_y = sigma
    return 1.0 - discrete_stieltjes(sigma_x, 1.0 / (z - sigma_y) ** 2, c)


def theta_discrete(c, sigma, tol):
    """Compute the effective signal."""
    sqrt_2_pi = np.sqrt(2 / np.pi)
    sigma_x,sigma_y=sigma
    sigma_image = [0, sigma_y[-1]]
    def f(lam):  # lam_max is the root of this function, note it decreases in lam
        def integrand(g):
            cdfs = norm.cdf(sigma_x - c * sqrt_2_pi * np.abs(g))
            cdfs_diff = cdfs[1:] - cdfs[:-1]
            return g**2 * (cdfs_diff / (lam - sigma_y)).sum()
        return gaussian_integral(integrand) - 1.0 / c

    a = sigma_image[1] + 1e-7  # lower bound for lam_max
    b = c + sigma_image[1] + 1e-7  # upper bound for lam_max
    if f(a) * f(b) > 0:  # no solution for f(lam) = 0
        return sigma_image[1]  # so that H'(theta) is -inf
    lam_max = spo.brentq(f, a, b, xtol=tol)
    return lam_max


def c_critical_discrete(c_range, sigma, plot=True, tol=2e-12):
    """Critical value of c."""
    if plot:  # plot the diagram for theta(c) and H_prime(theta(c)) values
        c_values = np.linspace(c_range[0], c_range[1], 100)
        theta_values = np.array([theta_discrete(c, sigma, tol) for c in c_values])
        H_prime_values = np.array(
            [H_prime_discrete(theta_values[i], sigma, c_values[i]) for i in range(len(c_values))]
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
        H_prime_discrete(theta_discrete(c_range[0], sigma, tol), sigma, c_range[0])
        * H_prime_discrete(theta_discrete(c_range[1], sigma, tol), sigma, c_range[1])
        > 0
    ):  # no solution for H'=0 in the range
        return c_range[1]
    c_critical = spo.brentq(
        lambda c: H_prime_discrete(theta_discrete(c, sigma, tol), sigma, c),
        c_range[0],
        c_range[1],
        xtol=tol,
    )
    return c_critical


def c_for_step_function(beta, c_range, plot=False, tol=2e-12):
    """
    Given parameter for the parametrized sigma function, compute c_critical.
    beta = (a[0], a[1], ..., a[n], b[0], ..., b[n])
    Define sigma to be a step function parametrized by:
    f(x) = y[i] for x in (x[i], x[i+1]) where
    x = (-np.inf, a[0], a[0:1].sum(), ..., a[0:n].sum(), np.inf)
    y = (0, b[0], b[0:1].sum(), ..., b[0:n].sum(), b[0:n].sum())
    """
    if isinstance(beta, list):
        beta = np.array(beta)
    x, y = beta_to_sigma(beta)
    c = c_critical_discrete(c_range, sigma=(x, y), plot=plot, tol=tol)
    print(c)
    return c
