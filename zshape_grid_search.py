import argparse
import pickle
import numpy as np
from tqdm import tqdm

import theoretical_analysis.planted_submatrix_analysis as submatrix
import theoretical_analysis.sparse_nnpca_analysis as gaussiannnpca

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["planted_submatrix", "gaussian_sparse_nnpca"])
    args = parser.parse_args()

    # Load the blackbox optimization result for z-shape sigma
    pickle_file = f"logs/{args.model}/optimization_zshape.pkl"
    try:
        with open(pickle_file, "rb") as f:
            res = pickle.load(f)
    except FileNotFoundError:
        print(f"File {pickle_file} not found. Please run the optimization first.")
        exit(1)

    # Define the computation of critical signal strength for the given model
    c_for_step_function = (
        submatrix.c_for_step_function
        if args.model == "planted_submatrix"
        else gaussiannnpca.c_for_step_function
    )
    c_critical = (
        submatrix.c_critical if args.model == "planted_submatrix" else gaussiannnpca.c_critical
    )

    # Define the range of parameters to explore
    a_range = np.arange(0, 14, 0.5)
    b_range = np.arange(0, 11, 0.5)
    c_range = np.arange(-10, 3, 0.5)

    # fix c, explore how c_critical changes with a,b
    c = res.x[-1]
    try:
        available_a, available_b, available_c_criticals = pickle.load(
            open(f"logs/{args.model}/c_criticals_c={c}.pkl", "rb")
        )
    except FileNotFoundError:
        available_a = []
        available_b = []
    c_criticals = np.zeros((len(a_range[:-1]), len(b_range[:-1])))

    pbar = tqdm(total=len(a_range[:-1]) * len(b_range[:-1]))
    for i, a in enumerate(a_range[:-1]):
        for j, b in enumerate(b_range[:-1]):
            if a in available_a[:-1] and b in available_b[:-1]:
                c_criticals[i, j] = available_c_criticals[
                    np.where(available_a == a)[0][0], np.where(available_b == b)[0][0]
                ]
                pbar.update(1)
                continue
            if a == 0:
                c_criticals[i, j] = c_for_step_function(
                    beta=[c, b], c_range=[0.5, 1.0], tol=1e-12, plot=False
                )
            else:
                c_criticals[i, j] = c_critical(
                    c_range=[0.5, 1.0],
                    sigma=lambda x: min(b, max(0, b / a * (x - c))),
                    sigma_image=[0, b],
                    tol=1e-12,
                    plot=False,
                )
            pbar.update(1)
    pbar.close()
    pickle.dump(
        [a_range, b_range, c_criticals], open(f"logs/{args.model}/c_criticals_c={c}.pkl", "wb")
    )

    # fix a, explore how c_critical changes with b,c
    a = res.x[0]
    try:
        available_b, available_c, available_c_criticals = pickle.load(
            open(f"logs/{args.model}/c_criticals_a={a}.pkl", "rb")
        )
    except:
        available_b = []
        available_c = []
    c_criticals = np.zeros((len(b_range[:-1]), len(c_range[:-1])))

    pbar = tqdm(total=len(b_range[:-1]) * len(c_range[:-1]))
    for i, b in enumerate(b_range[:-1]):
        for j, c in enumerate(c_range[:-1]):

            if b in available_b[:-1] and c in available_c[:-1]:
                c_criticals[i, j] = available_c_criticals[
                    np.where(available_b == b)[0][0], np.where(available_c == c)[0][0]
                ]
                pbar.update(1)
                continue

            c_criticals[i, j] = c_critical(
                c_range=[0.5, 1.0],
                sigma=lambda x: min(b, max(0, b / a * (x - c))),
                sigma_image=[0, b],
                tol=1e-12,
                plot=False,
            )
            pbar.update(1)
    pbar.close()
    pickle.dump(
        [b_range, c_range, c_criticals], open(f"logs/{args.model}/c_criticals_a={a}.pkl", "wb")
    )

    # fix b, explore how c_critical changes with a,c
    b = res.x[1]
    c_criticals = np.zeros((len(a_range[:-1]), len(c_range[:-1])))
    try:
        available_a, available_c, available_c_criticals = pickle.load(
            open(f"logs/{args.model}/c_criticals_b={b}.pkl", "rb")
        )
    except:
        available_a = []
        available_c = []
    pbar = tqdm(total=len(a_range[:-1]) * len(c_range[:-1]))
    for i, a in enumerate(a_range[:-1]):
        for j, c in enumerate(c_range[:-1]):
            if a in available_a[:-1] and c in available_c[:-1]:
                c_criticals[i, j] = available_c_criticals[
                    np.where(available_a == a)[0][0], np.where(available_c == c)[0][0]
                ]
                pbar.update(1)
                continue

            if a == 0:
                c_criticals[i, j] = c_for_step_function(
                    beta=[c, b], c_range=[0.5, 1.0], tol=1e-12, plot=False
                )
            else:
                c_criticals[i, j] = c_critical(
                    c_range=[0.5, 1.0],
                    sigma=lambda x: min(b, max(0, b / a * (x - c))),
                    sigma_image=[0, b],
                    tol=1e-12,
                    plot=False,
                )
            pbar.update(1)
    pbar.close()
    pickle.dump(
        [a_range, c_range, c_criticals], open(f"logs/{args.model}/c_criticals_b={b}.pkl", "wb")
    )
