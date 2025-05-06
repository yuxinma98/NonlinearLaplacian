import pickle
import glob
import os
import torch
import numpy as np
import wandb
from pathlib import Path
import argparse

from learning_from_data.train import NNTrainingModule
from theoretical_analysis.planted_submatrix_analysis import c_critical


def analyze_runs(task, model):
    """
    Analyze the runs for a given task and model.
    This function loads the runs from wandb, computes the c_critical value for each run,
    and saves the results to a pickle file.
    Args:
        task (str): The task name (e.g., "planted_submatrix").
        model (str): The model name (e.g., "relu").
    """
    api = wandb.Api()
    wandb_name = Path(os.environ["wandb_dir"])  # get wandb name from environment variable
    runs = api.runs(f"{wandb_name}/nonlinear_laplacian")  # get all runs from wandb logger
    checkpoint_dir = "learning_from_data/nonlinear_laplacian"  # directory where the checkpoints are saved
 
    try:
        results = pickle.load(open(f"logs/planted_submatrix/NN_{task}_{model}.pkl", "rb"))
    except FileNotFoundError:
        results = {}

    for run in runs:
        if not (run.config["task"] == task and run.config["model"]["model"] == model):
            continue
        if run.id in results:
            continue

        fname = glob.glob(f"{checkpoint_dir}/{run.id}/checkpoints/epoch*")[
            0
        ]  # get the best checkpoint
        best_model = NNTrainingModule.load_from_checkpoint(fname, map_location=None)
        best_model.eval()

        def learned_sigma(x):
            # post process of the model to enforce monotonicity
            if isinstance(x, (int, float)):
                input = torch.arange(x - 30, x, 0.01).reshape(-1, 1)
                input = input.float().to(best_model.device)
                output = best_model.model.mlp(input).detach().squeeze().cpu().numpy()
                return output.max()
            if isinstance(x, torch.Tensor) and x.dim() == 1:
                x_tensor = x.reshape(-1, 1).float().to(best_model.device)
                output = best_model.model.mlp(x_tensor).detach().squeeze().cpu().numpy()
                return np.maximum.accumulate(output)

        test_xs = torch.linspace(-100, 100, 1000)
        image = [learned_sigma(test_xs).min().item(), learned_sigma(test_xs).max().item()]
        c_critical_result = c_critical([0.7, 1.3], learned_sigma, sigma_image=image, plot=False)
        results[run.id] = [run.summary["test_acc/dataloader_idx_0"], c_critical_result]

        pickle.dump(results, open(f"logs/planted_submatrix/NN_{task}_{model}.pkl", "wb"))
        print(
            f"run_id={run.id}, test_acc={run.summary['test_acc/dataloader_idx_0']}, large_test_acc={run.summary['large_test_acc']}, c_critical={c_critical_result}"
        )


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument(
        "--task",
        type=str,
        choices=["planted_submatrix", "planted_clique"],
        default="planted_submatrix",
    )
    argparse.add_argument("--model", type=str, choices=["relu", "tanh"], default="tanh")
    args = argparse.parse_args()
    analyze_runs(args.task, args.model)
