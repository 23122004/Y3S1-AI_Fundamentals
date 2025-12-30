import numpy as np
from abc import ABC, abstractmethod

class Optimizer(ABC):
    """
    Abstract base class for continuous optimization algorithms.
    """
    def __init__(self, benchmark_func, max_iter, random_state=None):
        self.benchmark_func = benchmark_func
        self.max_iter = max_iter
        self.dim = self.benchmark_func.dim
        self.lower_bound, self.upper_bound = self.benchmark_func.get_bounds()
        
        # Set up random state
        self._random = self._initialize_random_state(random_state)
        
        # Attributes to store results
        self._best_solution = None
        self._best_fitness = np.inf
        self._convergence_history = []
        self._solution_history = []
        self._population_history = []

    @abstractmethod
    def solve(self):
        """
        Solve the optimization problem.
        """
        pass
    
    def _initialize_random_state(self, random_state):
        """
        Class method to initialize the random state.
        """
        if random_state is None:
            return np.random.RandomState()
        elif isinstance(random_state, int):
            return np.random.RandomState(random_state)
        elif isinstance(random_state, np.random.RandomState):
            return random_state
        else:
            raise ValueError("random_state must be None, int, or numpy.random.RandomState instance")

    def get_bounds(self):
        """
        Get the lower and upper bounds of the search space.
        """
        return self.lower_bound, self.upper_bound

    # Properties to access results
    @property
    def best_solution(self):
        return self._best_solution

    @property
    def best_fitness(self):
        return self._best_fitness

    @property
    def convergence_history(self):
        return self._convergence_history

    @property
    def solution_history(self):
        return self._solution_history

    @property
    def population_history(self):
        return self._population_history
