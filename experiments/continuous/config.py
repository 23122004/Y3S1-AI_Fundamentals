import numpy as np
import sys
from pathlib import Path

# Định nghĩa các đường dẫn quan trọng
CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = PROJECT_ROOT / "plots"
VISUAL_DIR = PROJECT_ROOT / "visualizations"

# Thêm project root vào Python path để import được module src
sys.path.insert(0, str(PROJECT_ROOT))

# Tạo thư mục nếu chưa tồn tại
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)
VISUAL_DIR.mkdir(exist_ok=True)

# Import algorithms from src
from src.algorithms.continuous.hill_climbing import HillClimbing
from src.algorithms.continuous.simulated_annealing import SimulatedAnnealing
from src.algorithms.continuous.artificial_bee_colony import ArtificialBeeColony
from src.algorithms.continuous.cuckoo_search import CuckooSearch
from src.algorithms.continuous.firefly_algorithm import FireflyAlgorithm

# Import benchmarks from same package
from benchmarks import Sphere, Rastrigin

# Import visualizer
from src.utils.convergence_visualizer import ConvergenceVisualizer

# === setup chung ===
N_RUNS = 30
MAX_ITER = 100
POP_SIZE = 40
DEFAULT_DIMENSIONS = 10
BENCHMARKS = {
    "Sphere": Sphere,
    "Rastrigin": Rastrigin
}

CS_DEFAULTS = {
    'population_size': POP_SIZE,
    'pa': 0.75,
    'alpha': 0.1,
    'beta': 1.5
}

ABC_DEFAULTS = {
    'population_size': POP_SIZE,
    'limit': (POP_SIZE // 2) * DEFAULT_DIMENSIONS
}

FA_DEFAULTS = {
    'population_size': POP_SIZE,
    'alpha': 0.1,
    'beta0': 2.0,
    'gamma': 0.1,
    'beta': 1.5
}

# === Setup cho phân tích độ nhạy tham số ===
SENSITIVITY_RESULTS_FILE = RESULTS_DIR / 'sensitivity_results.csv'

# Dải tham số cần kiểm tra cho CS
CS_SENSITIVITY = {
    'pa': [0.01, 0.1, 0.25, 0.5, 0.75],
    'alpha': [0.01, 0.1, 0.5, 1.0, 2.0]
}

# Dải tham số cần kiểm tra cho FA
FA_SENSITIVITY = {
    'alpha': [0.1, 0.2, 0.5, 0.8, 1.0],
    'beta0': [0.1, 0.5, 1.0, 1.5, 2.0],
    'gamma': [0.1, 0.5, 1.0, 2.0, 3.0]
}

# Dải tham số cần kiểm tra cho ABC
ABC_SENSITIVITY = {
    'limit': [5, 10, 20, 50, 100, 200, 300]
}

EXPERIMENTS = {
    'CuckooSearch': {
        'class': CuckooSearch,
        'defaults': CS_DEFAULTS,
        'params_to_test': CS_SENSITIVITY
    },
    'FireflyAlgorithm': {
        'class': FireflyAlgorithm,
        'defaults': FA_DEFAULTS,
        'params_to_test': FA_SENSITIVITY
    },
    'ArtificialBeeColony': {
        'class': ArtificialBeeColony,
        'defaults': ABC_DEFAULTS,
        'params_to_test': ABC_SENSITIVITY
    }
}

# === Setup cho so sánh thuật toán ===
DIMENSIONS = [5, 10, 15, 20, 25, 30]

# Tên tệp để lưu kết quả
RESULTS_FILE = RESULTS_DIR / 'results.csv'

# Định nghĩa các thuật toán và tham số của thuật toán
ALGORITHMS = {
    "HillClimbing": {
        "class": HillClimbing,
        "params": {"step_size": 0.1}
    },
    "SimulatedAnnealing": {
        "class": SimulatedAnnealing,
        "params": {"step_size": 0.1, "initial_temp": 1000.0, "cooling_rate": 0.95}
    },
    "ArtificialBeeColony": {
        "class": ArtificialBeeColony,
        "params": {**ABC_DEFAULTS}
    },
    "CuckooSearch": {
        "class": CuckooSearch,
        "params": {**CS_DEFAULTS}
    },
    "FireflyAlgorithm": {
        "class": FireflyAlgorithm,
        "params": {**FA_DEFAULTS}
    }
}

