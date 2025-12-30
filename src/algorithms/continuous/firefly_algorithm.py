from .base import Optimizer
import numpy as np
import math

class FireflyAlgorithm(Optimizer):
    def __init__(self, benchmark_func, max_iter, random_state=None, 
                 population_size=50, alpha=0.5, beta0=0.2, gamma=1.0, 
                 beta=1.5):
        
        super().__init__(benchmark_func, max_iter, random_state)
        self.population_size = population_size
        self.alpha = alpha
        self.beta0 = beta0 
        self.gamma = gamma
        self.beta = beta

        # Initialize population
        self._fireflies = self._random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self._fitness = np.array([self.benchmark_func.evaluate(firefly) for firefly in self._fireflies])

        # Find initial best firefly
        best_firefly_id = np.argmin(self._fitness)
        self._best_fitness = self._fitness[best_firefly_id]
        self._best_solution = self._fireflies[best_firefly_id].copy()
        
        # Convergence history
        self._convergence_history.append(self._best_fitness)
        self._solution_history.append(self._best_solution)
        self._population_history.append(self._fireflies.copy())

    def solve(self):
        """
        Main loop of the Firefly Algorithm.
        """
        for _ in range(self.max_iter):
            for i in range(self.population_size):
                for j in range(self.population_size):
                    if self._fitness[j] < self._fitness[i]:
                        r_squared = np.sum((self._fireflies[i] - self._fireflies[j])**2)
                        attractiveness = self.beta0 * np.exp(-self.gamma * r_squared)
                        random_step = self.alpha * self._generate_levy_step()
                        
                        # Move i towards j based on attractiveness
                        self._fireflies[i] += attractiveness * (self._fireflies[j] - self._fireflies[i]) + random_step
                        self._fireflies[i] = np.clip(self._fireflies[i], self.lower_bound, self.upper_bound)
                        self._fitness[i] = self.benchmark_func.evaluate(self._fireflies[i])
                        
                        # Update best solution
                        if self._fitness[i] < self._best_fitness:
                            self._best_fitness = self._fitness[i]
                            self._best_solution = self._fireflies[i].copy()

            self._convergence_history.append(self._best_fitness)
            self._solution_history.append(self._best_solution)
            self._population_history.append(self._fireflies.copy())

        return self._best_solution, self._best_fitness

    def _generate_levy_step(self):
        """
        Tạo một bước nhảy Lévy (logic từ Cuckoo Search)
        """
        beta = self.beta
        sigma_u = (math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        sigma_v = 1
        u = self._random.normal(0, sigma_u, self.dim)
        v = self._random.normal(0, sigma_v, self.dim)
        levy_step = u / (np.abs(v)**(1 / beta))
        return levy_step
