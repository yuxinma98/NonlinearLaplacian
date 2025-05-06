from pathlib import Path
import os

data_dir = Path(os.environ["DATA_DIR"]) / "custom_laplacian"
CURRENT_DIR = Path(__file__).resolve().parent
