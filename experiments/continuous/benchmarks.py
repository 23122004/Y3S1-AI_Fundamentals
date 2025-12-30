import numpy as np
from abc import ABC, abstractmethod

class BenchmarkFunction(ABC):
    """
    Abstract base class for continuous benchmark functions.
    """
    def __init__(self, dim, lower_bound, upper_bound):
        self.dim = dim
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    @abstractmethod
    def evaluate(self, x):
        pass

    def get_bounds(self):
        return self.lower_bound * np.ones(self.dim), self.upper_bound * np.ones(self.dim)

class Sphere(BenchmarkFunction):
    def __init__(self, dim, lower_bound=-10.0, upper_bound=10.0):
        super().__init__(dim, lower_bound, upper_bound)

    def evaluate(self, x):
        return np.sum(x**2)

class Rastrigin(BenchmarkFunction):
    def __init__(self, dim, lower_bound=-5.12, upper_bound=5.12):
        super().__init__(dim, lower_bound, upper_bound)

    def evaluate(self, x):
        return 10 * self.dim + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))