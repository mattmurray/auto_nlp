import torch
from pathlib import Path

# BASE_PATH = Path.cwd()
BASE_PATH = Path(r"C:\Users\mattm\PycharmProjects\nlp_ml_template")
OUTPUT_PATH = BASE_PATH / 'output'
INPUT_PATH = BASE_PATH / 'input'
SRC_PATH = BASE_PATH / 'src'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
