import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
import scipy.optimize as spo
from scipy.stats import norm
from theoretical_analysis import gaussian_integral, beta_to_sigma


def generate_planted_matrix(n, beta):
    """Generate a matrix with a planted submatrix."""
    k = int(np.sqrt(n) * beta)
    A_p = np.random.randn(n, n)
    A_p = (A_p + A_p.T) / np.sqrt(2)
    clique_vertices = np.random.choice(n, k, replace=False)
    A_p[clique_vertices.reshape(-1, 1), clique_vertices.reshape(1, -1)] += 1
    y = np.zeros(n)
    y[clique_vertices] = 1
    if np.linalg.norm(y) > 0:
        y /= np.linalg.norm(y)
    return (A_p / np.sqrt(n), y)


def compute_free_convolution(sigma, zs_x, z_imag_part=1e-4, tolerance=1e-5, whole_support=True):
    """Compute the esd, i.e. free convolution of \mu_{SC} \boxplus \sigma(N(0,1))"""
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
    lam_max = spo.brentq(f, a, b, xtol=tol)
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
    c_critical = spo.brentq(
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
    """H function. H(z) = z + G_{\nu}(z). \nu = \sigma(N(0,1)) in this case."""
    sigma_x, sigma_y = sigma
    return z + gaussian_integral_discrete(sigma_x, 1.0 / (z - sigma_y))


def H_prime_discrete(z, sigma):
    """H function derivative. H'(z) = 1 + G_{\nu}'(z)."""
    if (
        z >= 0 and z <= sigma[1][-1]
    ):  # H_prime undefined when z is in the range of the support of nu
        return -np.inf
    sigma_x, sigma_y = sigma
    return 1.0 - gaussian_integral_discrete(sigma_x, 1.0 / (z - sigma_y) ** 2)


def theta_discrete(c, sigma, tol, **kwargs):
    """Compute the effective signal: the largest eigenvalue of signal X."""
    sigma_x, sigma_y = sigma
    sigma_image = [0, sigma_y[-1]]

    def f(lam):  # lam_max is the root of this function, note it decreases in lam
        x = sigma_x - c
        y = 1.0 / (lam - sigma_y)
        return gaussian_integral_discrete(x, y) - 1.0 / c

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
            [
                H_prime_discrete(
                    theta,
                    sigma,
                )
                for theta in theta_values
            ]
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
        H_prime_discrete(theta_discrete(c_range[0], sigma, tol), sigma)
        * H_prime_discrete(theta_discrete(c_range[1], sigma, tol), sigma)
        > 0
    ):  # no solution for H'=0 in the range
        return c_range[1]
    c_critical = spo.brentq(
        lambda c: H_prime_discrete(theta_discrete(c, sigma, tol), sigma),
        c_range[0],
        c_range[1],
        xtol=tol,
    )
    return c_critical


def c_for_step_function(beta, c_range, plot=False, tol=2e-12):
    if isinstance(beta, list):
        beta = np.array(beta)
    x, y = beta_to_sigma(beta)
    c = c_critical_discrete(c_range, sigma=(x, y), plot=plot, tol=tol)
    return c
