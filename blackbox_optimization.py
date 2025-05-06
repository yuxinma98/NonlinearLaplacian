import os
import argparse
import pickle
import scipy.optimize as spo
import numpy as np

import theoretical_analysis.planted_submatrix_analysis as submatrix
import theoretical_analysis.sparse_nnpca_analysis as gaussiannnpca

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["planted_submatrix", "gaussian_sparse_nnpca"])
    parser.add_argument("--sigma", type=str, choices=["zshape", "tanh", "step"])
    args = parser.parse_args()

    pickle_file = f"logs/{args.model}/optimization_{args.sigma}.pkl"
    os.makedirs(os.path.dirname(pickle_file), exist_ok=True)
    if os.path.exists(pickle_file):
        results = pickle.load(open(pickle_file, "rb"))
        if (args.sigma == "step" and len(results) == 10) or (args.sigma != "step"):
            print(f"Optimization result already exists. Check out {pickle_file}.")
            exit(0)
    else:
        results = {}

    print(f"Running optimization...")

    c_critical = (
        submatrix.c_critical if args.model == "planted_submatrix" else gaussiannnpca.c_critical
    )
    c_for_step_function = (
        submatrix.c_for_step_function
        if args.model == "planted_submatrix"
        else gaussiannnpca.c_for_step_function
    )

    if args.sigma == "zshape":

        def find_c_for_parametrized_sigma(gamma):
            """Given parameter for the parametrized sigma function, compute c_critical."""
            sigma = lambda x: min(gamma[1], max(0, gamma[1] / gamma[0] * (x - gamma[2])))
            sigma_image = [0, gamma[1]]
            c = c_critical([0.6, 1.0], sigma=sigma, sigma_image=sigma_image, plot=False)
            print(f"parameters={gamma}, critical signal strength={c}")
            return c

        # run Nelder-Mead optimization
        res = spo.minimize(
            find_c_for_parametrized_sigma,
            x0=[2, 2, -1],  # initial guess for the parameters
            method="Nelder-Mead",
            # options={"xatol": 0.1, "fatol": 1e-5, "disp": True},
        )
        with open(pickle_file, "wb") as f:
            pickle.dump(res, f)
    elif args.sigma == "tanh":

        def find_c_for_parametrized_sigma(gamma):
            """Given parameter for the parametrized sigma function, compute c_critical."""
            sigma = lambda x: gamma[0] * np.tanh(gamma[1] * x)
            sigma_image = [-gamma[0], gamma[0]]
            c = c_critical([0.6, 1.0], sigma=sigma, sigma_image=sigma_image, plot=False)
            print(f"parameters={gamma}, critical signal strength={c}")
            return c

        res = spo.minimize(
            find_c_for_parametrized_sigma,
            x0=[1, 1],  # initial guess for the parameters
            method="Nelder-Mead",
            # options={"xatol": 0.1, "fatol": 1e-5, "disp": True},
        )
        with open(pickle_file, "wb") as f:
            pickle.dump(res, f)

    elif args.sigma == "step":

        def minimize_with_random_initial_simplex(n=20, seed=0):
            # define bounds for the parameters in optimization
            bounds = (
                [(None, None)]
                + [(0, None) for _ in range(1, n + 1)]
                + [(0, None) for _ in range(0, n + 1)]
            )
            # define initial simplex for Nelder-Mead
            rng = np.random.default_rng(seed)
            N = 2 * n + 2  # N parameters, need N+1 points to define the initial simplex
            simplex = rng.uniform(
                0, 1, (N + 1, N)
            )  # set each incremental parameter to be uniformly random in [0,1]
            simplex[:, 0] = rng.uniform(
                -5, 0, N + 1
            )  # set x0(the first parameter) to be uniformly random in [-5,0]
            # run minimization
            historic_xks = []
            historic_funs = []

            def callback_function(intermediate_result):
                historic_xks.append(intermediate_result.x)
                historic_funs.append(intermediate_result.fun)

            res = spo.minimize(
                lambda beta: c_for_step_function(beta, c_range=[0.5, 1.5], tol=1e-12, plot=False),
                x0=np.zeros(2 * n + 2),  # will be overwritten by the simplex
                method="Nelder-Mead",
                bounds=bounds,
                callback=callback_function,
                options={"initial_simplex": simplex},
                # options={"initial_simplex": simplex, "xatol": 0.1, "fatol": 1e-5, "disp": True},
            )
            return historic_xks, historic_funs, res

        for seed in range(10):  # run the optimization for 10 different seeds
            if str(seed) in results:
                continue
            print(f"Running optimization for seed {seed}...")
            historic_xks, historic_funs, res = minimize_with_random_initial_simplex(n=15, seed=seed)
            results[str(seed)] = [historic_xks, historic_funs, res]
            with open(pickle_file, "wb") as f:
                pickle.dump(results, f)
