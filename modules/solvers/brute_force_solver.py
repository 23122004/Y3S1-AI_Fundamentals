import sys
import itertools
from modules.solvers.base_solver import BaseSolver

sys.setrecursionlimit(5000)

class BruteForceSolver(BaseSolver):
    def __init__(self, grid):
        super().__init__(grid)

    def solve(self):
        self._init_cnf()

        for state in itertools.product([False, True], repeat=self.num_vars):
            if self._check_all_clauses(state) is True:
                if self._check_connectivity(state):
                    return self._state_to_model(state)
        return None
