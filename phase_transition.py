import argparse
import pickle
import numpy as np
import torch
from tqdm import tqdm

import theoretical_analysis.planted_submatrix_analysis as submatrix
import theoretical_analysis.sparse_nnpca_analysis as gaussiannnpca
from theoretical_analysis import beta_to_sigma, step_function

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["planted_submatrix", "gaussian_sparse_nnpca"])
    parser.add_argument("--sigma", type=str, choices=["zshape", "tanh", "step"], default="tanh")
    args = parser.parse_args()

    # Load the blackbox optimization result for z-shape sigma
    pickle_file = f"logs/{args.model}/optimization_{args.sigma}.pkl"
    try:
        with open(pickle_file, "rb") as f:
            res = pickle.load(f)
    except FileNotFoundError:
        print(f"File {pickle_file} not found. Please run the optimization first.")
        exit(1)

    # set up
    n = 2000  # Dimension of the matrix
    N = 100  # Number of samples per beta
    np.random.seed(0)
    betas = np.arange(0, 3, 0.05)  # Range of beta values
    generator = (
        gaussiannnpca.generate_nnpca_matrix
        if args.model == "gaussian_sparse_nnpca"
        else submatrix.generate_planted_matrix
    )
    if args.sigma == "step":
        results_list = [res[str(seed)][2].fun for seed in range(10)]
        argmin_seed = np.argmin(results_list)
        res = res[str(argmin_seed)][2]

    # Load existing results if available
    try:
        results = pickle.load(
            open(f"logs/{args.model}/{args.sigma}_top_eigen_n={n}_N={N}.pkl", "rb")
        )
    except FileNotFoundError:
        results = {}

    # Computation of top eigenvalues and eigenvectors
    for beta in betas:
        if beta in results:
            continue
        A_p_batch = torch.zeros((N, n, n))
        y_batch = torch.zeros((N, n))

        for i in range(N):
            A_p, y = generator(n, beta)
            A_p_batch[i] = torch.tensor(A_p)
            y_batch[i] = torch.tensor(y)

        evals, evecs = torch.linalg.eigh(A_p_batch)  # Batch eigenvalue decomposition
        A_p_evals = evals[:, -1].numpy()  # Largest eigenvalues
        A_p_evecs = (
            (evecs[:, :, -1] * y_batch).sum(dim=1) ** 2
        ).numpy()  # Corresponding eigenvectors

        # Compute L_p in batch
        if args.sigma == "tanh":
            diag_values = res.x[0] * torch.tanh(res.x[1] * A_p_batch.sum(dim=2))
        elif args.sigma == "zshape":
            diag_values = torch.clamp(
                res.x[1] / res.x[0] * (A_p_batch.sum(dim=2) - res.x[2]), min=0, max=res.x[1]
            )
        elif args.sigma == "step":
            sigma = beta_to_sigma(res.x)
            diag_values = step_function(sigma, A_p_batch.sum(dim=2))
        L_p_batch = A_p_batch + torch.diag_embed(diag_values)
        evals, evecs = torch.linalg.eigh(L_p_batch)
        L_p_evals = evals[:, -1].numpy()
        L_p_evecs = ((evecs[:, :, -1] * y_batch).sum(dim=1) ** 2).numpy()

        results[beta] = [A_p_evals, A_p_evecs, L_p_evals, L_p_evecs]
        print(
            f"beta: {beta:.2f}, A_p_evals: {A_p_evals.mean():.4f}, L_p_evals: {L_p_evals.mean():.4f}"
        )
        pickle.dump(
            results, open(f"logs/{args.model}/{args.sigma}_top_eigen_n={n}_N={N}.pkl", "wb")
        )
