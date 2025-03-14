import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
import scipy.optimize as spo
from scipy.stats import norm


def gaussian_integral(f):
    """General calculation of E[f(x)] for x ~ N(0, 1)."""
    return spi.quad(
        lambda x: f(x) * np.exp(-(x**2) / 2.0) / np.sqrt(2.0 * np.pi), -np.inf, np.inf
    )[0]


def H(z, sigma):
    """H function. H(z) = z + G_{\nu}(z). \nu = \sigma(N(0,1)) in this case."""
    return z + gaussian_integral(lambda g: 1 / (z - sigma(g)))


def H_prime(z, sigma, sigma_image):
    """H function derivative. H'(z) = 1 + G_{\nu}'(z)."""
    if (
        z >= sigma_image[0] and z <= sigma_image[1]
    ):  # H_prime undefined when z is in the range of the support of nu
        return -np.inf
    return 1.0 - gaussian_integral(lambda g: 1 / (z - sigma(g)) ** 2)


def theta(c, sigma, sigma_image, tol):
    """Compute the effective signal: the largest eigenvalue of signal X."""

    def f(lam):  # lam_max is the root of this function, note it is strictly decreasing in lam
        return gaussian_integral(lambda g: 1.0 / (lam - sigma(c + g))) - 1.0 / c

    a = sigma_image[1] + 1e-7  # lower bound for lam_max
    b = c + sigma_image[1] + 1e-7  # upper bound for lam_max
    if f(a) * f(b) > 0:  # no solution for f(lam) = 0
        return sigma_image[1]  # so that H'(theta) is inf
    lam_max = spo.bisect(f, a, b, xtol=tol)
    return lam_max


def c_critical(c_range, sigma, sigma_image, plot=True, tol=2e-12):
    """Find critical value of c by solving H'(theta(c,sigma)) = 0."""
    if plot:  # plot the diagram for theta(c) and H_prime(theta(c)) values
        c_values = np.linspace(c_range[0], c_range[1], 100)
        theta_values = np.array([theta(c, sigma, sigma_image, tol=tol) for c in c_values])
        H_prime_values = np.array([H_prime(theta, sigma, sigma_image) for theta in theta_values])
        # i = np.where(H_prime_values > -1)[0][0] # only plot the part where H_prime > -1
        i = 0
        fig, axes = plt.subplots(2, 1, sharex=True)
        axes[0].plot(c_values[i:], theta_values[i:])
        axes[1].plot(c_values[i:], H_prime_values[i:])
        axes[0].set_xlabel("$c$ (clique size)")
        axes[0].set_ylabel("$\\theta(c)$ (effective signal)")
        axes[1].set_xlabel("$c$ (clique size)")
        axes[1].set_ylabel("$H^{\\prime}(\\theta(c))$")
        plt.show()
        plt.close()

    if (
        H_prime(theta(c_range[0], sigma, sigma_image, tol=tol), sigma, sigma_image)
        * H_prime(theta(c_range[1], sigma, sigma_image, tol=tol), sigma, sigma_image)
        > 0
    ):  # no solution for H'=0 in the range
        return c_range[1]
    c_critical = spo.bisect(
        lambda c: H_prime(theta(c, sigma, sigma_image, tol), sigma, sigma_image),
        c_range[0],
        c_range[1],
        xtol=tol,
    )
    return c_critical


def outlier_eval(c, sigma, sigma_image, tol=2e-12):
    """Evaluate the outlier eigenvalue when c > c_critical."""
    return H(theta(c, sigma, sigma_image, tol), sigma)


########################## Discrete version for fast evaluation on step functions #########################


def gaussian_integral_discrete(x, y):
    """General calculation of E[f(x)] for x ~ N(0, 1)."""
    """input: x, y are np arrays define a step function f(x) = y[i] for x in (x[i], x[i+1])"""
    assert x[0] == -np.inf and x[-1] == np.inf
    F_x = norm.cdf(x)
    F_x_diff = F_x[1:] - F_x[:-1]
    return (F_x_diff * y).sum()


def H_discrete(z, sigma):
    """H function."""
    sigma_x, sigma_y = sigma
    return z + gaussian_integral_discrete(sigma_x, 1.0 / (z - sigma_y))


def H_prime_discrete(z, sigma, sigma_image):
    """H function derivative."""
    if (
        z >= sigma_image[0] and z <= sigma_image[1]
    ):  # H_prime undefined when z is in the range of the support of nu
        return -np.inf
    sigma_x, sigma_y = sigma
    return 1.0 - gaussian_integral_discrete(sigma_x, 1.0 / (z - sigma_y) ** 2)


def theta_discrete(c, sigma, sigma_image, tol):
    """Compute the effective signal: the largest eigenvalue of signal X."""
    sigma_x, sigma_y = sigma

    def f(lam):  # lam_max is the root of this function, note it decreases in lam
        x = sigma_x - c
        y = 1.0 / (lam - sigma_y)
        return gaussian_integral_discrete(x, y) - 1.0 / c

    a = sigma_image[1] + 1e-7  # lower bound for lam_max
    b = c + sigma_image[1] + 1e-7  # upper bound for lam_max
    if f(a) * f(b) > 0:  # no solution for f(lam) = 0
        return sigma_image[1]  # so that H'(theta) is -inf
    lam_max = spo.bisect(f, a, b, xtol=tol)
    return lam_max


def c_critical_discrete(c_range, sigma, sigma_image, plot=True, tol=2e-12):
    """Critical value of c."""
    if plot:  # plot the diagram for theta(c) and H_prime(theta(c)) values
        c_values = np.linspace(c_range[0], c_range[1], 100)
        theta_values = np.array([theta_discrete(c, sigma, sigma_image, tol) for c in c_values])
        H_prime_values = np.array(
            [H_prime_discrete(theta, sigma, sigma_image) for theta in theta_values]
        )
        i = np.where(H_prime_values > -1)[0][0]  # only plot the part where H_prime > -1
        fig, axes = plt.subplots(2, 1, sharex=True)
        axes[0].plot(c_values[i:], theta_values[i:])
        axes[1].plot(c_values[i:], H_prime_values[i:])
        axes[0].set_xlabel("$c$ (clique size)")
        axes[0].set_ylabel("$\\theta(c)$ (effective signal)")
        axes[1].set_xlabel("$c$ (clique size)")
        axes[1].set_ylabel("$H^{\\prime}(\\theta(c))$")
        plt.show()
        plt.close()
    if (
        H_prime_discrete(theta_discrete(c_range[0], sigma, sigma_image, tol), sigma, sigma_image)
        * H_prime_discrete(theta_discrete(c_range[1], sigma, sigma_image, tol), sigma, sigma_image)
        > 0
    ):  # no solution for H'=0 in the range
        return c_range[1]
    c_critical = spo.bisect(
        lambda c: H_prime_discrete(theta_discrete(c, sigma, sigma_image, tol), sigma, sigma_image),
        c_range[0],
        c_range[1],
        xtol=tol,
    )
    return c_critical


def plot_step_function(beta):
    n = len(beta) // 2 - 1
    a, b = beta[0 : n + 1], beta[n + 1 :]
    x = np.cumsum(a)
    x = np.concatenate(([-np.inf], x, [np.inf]))
    y = np.cumsum(b)
    y = np.concatenate(([0], y))
    x_s = np.linspace(-10, 10, 1000)
    y_s = np.zeros_like(x_s)
    for i in range(0, n + 2):
        y_s[x_s >= x[i]] = y[i]
    plt.plot(x_s, y_s)
    plt.show()
    plt.close()


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
    n = len(beta) // 2 - 1
    a, b = beta[0 : n + 1], beta[n + 1 :]
    assert (a[1:] >= 0).all()
    assert (b >= 0).all()
    x = np.cumsum(a)
    x = np.concatenate(([-np.inf], x, [np.inf]))
    y = np.cumsum(b)
    y = np.concatenate(([0], y))

    image = [0, b.sum()]
    c = c_critical_discrete(c_range, sigma=(x, y), sigma_image=image, plot=plot, tol=tol)
    return c


def generate_planted_matrix(n, k):
    """Generate a matrix with a planted submatrix."""
    A_p = np.random.randn(n, n)
    A_p = (A_p + A_p.T) / np.sqrt(2)
    clique_vertices = np.random.choice(n, k, replace=False)
    A_p[clique_vertices.reshape(-1, 1), clique_vertices.reshape(1, -1)] += 1
    return A_p / np.sqrt(n)


def compute_free_convolution(sigma, zs_x, z_imag_part=1e-4, tolerance=1e-5, whole_support=True):
    """Compute the free convolution of \mu_{SC} \boxplus \sigma(N(0,1))"""
    zs = np.hstack([zs_x + z_imag_part * 1.0j])
    num_zs = zs_x.shape[0]
    # recursion for G
    G = np.zeros((num_zs), dtype=np.complex128)
    while True:
        G_old = G.copy()
        for j in range(num_zs):
            real_integrand = lambda g: np.real(
                1 / (np.sqrt(2 * np.pi)) * np.exp(-(g**2) / 2) / (zs[j] - G[j] - sigma(g))
            )
            imag_integrand = lambda g: np.imag(
                1 / (np.sqrt(2 * np.pi)) * np.exp(-(g**2) / 2) / (zs[j] - G[j] - sigma(g))
            )

            real_result, real_error = spi.quad(real_integrand, -np.inf, np.inf)
            imag_result, imag_error = spi.quad(imag_integrand, -np.inf, np.inf)

            G[j] = real_result + 1j * imag_result
        if np.linalg.norm(G - G_old) < tolerance:
            break

    # Stieltjes inversion
    rho = -1 / np.pi * np.imag(G)
    if whole_support and (rho[0] > tolerance or rho[-1] > tolerance):
        print(rho[0], rho[-1])
        raise Exception(
            "Warning: zs_x does not cover the whole support. Increase the range of zs_x."
        )
    return rho
