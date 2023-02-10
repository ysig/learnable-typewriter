"""Defines Constants"""
from pathlib import Path
import os

# Project and source files
PROJECT_PATH = Path(__file__).parent.parent.parent.resolve()
CONFIGS_PATH = PROJECT_PATH / 'configs'
DATASETS_PATH = PROJECT_PATH / 'datasets'
OUTPUT_PATH = Path(os.environ['SCRATCH']) if 'SCRATCH' in os.environ else PROJECT_PATH
RUNS_PATH = OUTPUT_PATH / 'runs'
RESULTS_PATH = OUTPUT_PATH / 'results'
MODEL_FILE = 'model.pth'
BEST_MODEL = 'model_best.pth'
