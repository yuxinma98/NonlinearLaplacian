import argparse
from learning_from_data.train import train
from learning_from_data import CURRENT_DIR


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="tanh",
        help="Neural network model choice",
        choices=["tanh", "relu"],
    )
    parser.add_argument(
        "--task",
        type=str,
        default="planted_submatrix",
        help="Task to perform",
        choices=["planted_submatrix", "nonnegative_pca", "planted_clique"],
    )
    parser.add_argument("--max_epochs", type=int, default=350, help="Number of epochs to train")
    args = parser.parse_args()
    params = {
        "project": "custom_laplacian",
        "name": f"{args.task}_{args.model}",
        "log_dir": CURRENT_DIR,
        # --------data parameters--------
        "task": args.task,
        "data_seed": 42,
        "batch_size": 200,
        "N": 5000,  # number of instances
        "n": 100,  # matrix size
        "beta": 1.3,  # signal-to-noise ratio
        "test_N": 100,  # number of test instances
        "test_n": 2000,  # test matrix size
        "test_fraction": 0.3,
        "val_fraction": 0.1,
        # --------training parameters--------
        "model": {
            "model": args.model,
            "num_layers": 5,
            "hidden_channels": 30,
        },
        "max_epochs": args.max_epochs,
        "weight_decay": 0.01,
        "lr": 0.01,
        "lr_patience": 10,
        # --------logging parameters--------
        "logger": True,  # whether to use logger at all (wandb)
        "log_checkpoint": True,  # whether to log model checkpoints: True (log last & best checkpoint), False or "all" (log all checkpoints)
        "log_model": "all",  # "gradient", "parameters", "all", False
    }
    for i in range(10):
        params["training_seed"] = i
        train(params)
