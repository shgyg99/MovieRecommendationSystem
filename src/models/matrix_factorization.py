import numpy as np
import pandas as pd
from collections import defaultdict
import pickle
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.config_manager import config_manager



if __name__ == "__main__":

            
        