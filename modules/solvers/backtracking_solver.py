import sys
from modules.solvers.base_solver import BaseSolver

sys.setrecursionlimit(5000)

class BacktrackingSolver(BaseSolver):
    def __init__(self, grid):
        super().__init__(grid)
        self.state = []  # state[state_idx] -> True/False/None

    def solve(self):
        self._init_cnf()

        # Initialize state list (state_idx = var_id - 1)
        self.state = [None] * self.num_vars

        if self._backtrack(0):
            return self._state_to_model(self.state)

        return None

    def _backtrack(self, state_idx):
        result = self._check_all_clauses(self.state)
        if result is False:
            return False

        if state_idx == self.num_vars:
            return self._check_connectivity(self.state)

        # Try True
        self.state[state_idx] = True
        if self._backtrack(state_idx + 1):
            return True

        # Try False
        self.state[state_idx] = False
        if self._backtrack(state_idx + 1):
            return True

        # Backtrack
        self.state[state_idx] = None
        return False
