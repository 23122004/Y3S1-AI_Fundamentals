from modules.solvers.pysat_solver import PySatSolver
from modules.solvers.astar_solver import AStarSolver
from modules.solvers.backtracking_solver import BacktrackingSolver
from modules.solvers.brute_force_solver import BruteForceSolver

NUM_TESTCASE = 30

ALGORITHMS = {
    'pysat': PySatSolver,
    'astar': AStarSolver,
    'backtracking': BacktrackingSolver,
    'brute_force': BruteForceSolver
}

TIMEOUT = 60  # seconds

TEST_SIZES = [7, 9, 11, 13, 17, 20]