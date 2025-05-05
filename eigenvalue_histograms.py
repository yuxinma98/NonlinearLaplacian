import argparse
import pickle
import numpy as np
from theoretical_analysis import planted_submatrix_analysis as submatrix
from theoretical_analysis import beta_to_sigma, step_function

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma", type=str, choices=["zshape", "tanh", "step"], default="tanh")
    args = parser.parse_args()

    n = 10000
    np.random.seed(0)
    with open(f"logs/planted_submatrix/optimization_{args.sigma}.pkl", "rb") as f:
        res = pickle.load(f)

    # Load existing results if available
    try:
        results = pickle.load(
            open(f"logs/planted_submatrix/{args.sigma}_eigenvalues_n={n}.pkl", "rb")
        )
    except FileNotFoundError:
        results = {}
    for beta in [0, 0.9, 1.2]:
        if str(beta) in results:
            continue
        A_p, _ = submatrix.generate_planted_matrix(n, int(np.sqrt(n) * beta))
        A_p_evals = np.linalg.eigvalsh(A_p)
        if args.sigma == "tanh":
            diag_values = res.x[0] * np.tanh(res.x[1] * A_p.sum(axis=1))
            sigma = lambda x: res.x[0] * np.tanh(res.x[1] * x)
        elif args.sigma == "zshape":
            diag_values = np.clamp(
                res.x[1] / res.x[0] * (A_p.sum(axis=1)) - res.x[2], min=0, max=res.x[1]
            )
            sigma = lambda x: min(res.x[1], max(0, res.x[1] / res.x[0] * (x - res.x[2])))
        elif args.sigma == "step":
            sigma = beta_to_sigma(res.x)
            diag_values = step_function(sigma, A_p.sum(axis=1))
            sigma = lambda x: step_function(sigma, x)

        L_p = A_p + np.diag(diag_values)
        L_p_evals = np.linalg.eigvalsh(L_p)

        # Compute the free convolution
        x = np.linspace(-3, 3, 500)
        sc = np.sqrt(np.maximum(4 - x**2, 0)) / (2 * np.pi)
        free_conv = submatrix.compute_free_convolution(sigma=sigma, zs_x=x)

        results[str(beta)] = [A_p_evals, L_p_evals, x, sc, free_conv]
        pickle.dump(
            results,
            open(f"logs/planted_submatrix/eigenvalues.pkl", "wb"),
        )
