import torch
from pathlib import Path

# BASE_PATH = Path.cwd()
BASE_PATH = Path(r"C:\Users\mattm\PycharmProjects\nlp_ml_template")
OUTPUT_PATH = BASE_PATH / 'output'
# NOTEBOOKS_PATH = BASE_PATH / 'notebooks'
INPUT_PATH = BASE_PATH / 'input'
SRC_PATH = BASE_PATH / 'src'
# ASSETS_PATH = BASE_PATH / 'assets'

# DISABLE_CV_FOR_DL_MODELS = True
# RANDOM_STATE = 42
N_JOBS = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
