from .base import Optimizer
import numpy as np

class CuckooSearch(Optimizer):
    def __init__(self, 
                 benchmark_func, 
                 max_iter, 
                 population_size=50,
                 random_state=None, 
                 alpha=0.01, 
                 beta=1.5, 
                 pa=0.25
                 ):
        super().__init__(benchmark_func, max_iter, random_state)
        self.population_size = population_size
        self.alpha = alpha
        self.beta = beta
        self.pa = pa

        # Initialize population
        self._nests = self._random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self._fitness = np.array([self.benchmark_func.evaluate(nest) for nest in self._nests])

        # Initial best and solution
        best_nest_id = np.argmin(self._fitness)
        self._best_fitness = self._fitness[best_nest_id]
        self._best_solution = self._nests[best_nest_id].copy()

        # Convergence history   
        self._convergence_history.append(self._best_fitness)
        self._solution_history.append(self._best_solution)
        self._population_history.append(self._nests.copy())

    def solve(self):
        """
        Main loop of Cuckoo Search algorithm
        """
        for _ in range(self.max_iter):
            # Replace a nests randomly with Lévy flight
            self._levy_flight()

            # Abandon a fraction of worse nests and build new ones
            self._abandon_nests()
            
            # Update the current best
            curr_best_id = np.argmin(self._fitness)
            if self._fitness[curr_best_id] < self._best_fitness:
                self._best_fitness = self._fitness[curr_best_id]
                self._best_solution = self._nests[curr_best_id].copy()

            self._convergence_history.append(self._best_fitness)
            self._solution_history.append(self._best_solution)
            self._population_history.append(self._nests.copy())
            
        return self._best_solution, self._best_fitness

    def _levy_flight(self):
        # Generate a new solution using Lévy flight
        x_i = self._nests[self._random.randint(0, self.population_size)]
        new_nest = self._create_nest(x_i)
        new_fitness = self.benchmark_func.evaluate(new_nest)

        # Select a random nest to replace
        j = self._random.randint(0, self.population_size)
        
        # Replace if better
        if new_fitness < self._fitness[j]:
            self._nests[j] = new_nest
            self._fitness[j] = new_fitness

    def _create_nest(self, nest):
        # Levy flight parameters
        beta = self.beta
        sigma_u = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (np.math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        sigma_v = 1

        # Generate Levy step
        u = self._random.normal(0, sigma_u, self.dim)
        v = self._random.normal(0, sigma_v, self.dim)
        levy_step = u / (np.abs(v)**(1 / beta))

        # New solution
        new_nest = nest + self.alpha  * levy_step
        new_nest = np.clip(new_nest, self.lower_bound, self.upper_bound)
        return new_nest

    def _abandon_nests(self):
        num_abandoned = int(self.pa * self.population_size)
        worst_indices = np.argsort(self._fitness)[-num_abandoned:]

        # Create new solutions for abandoned nests
        for i in worst_indices:
            j, k = self._random.choice(self.population_size, 2, replace=False)
            
            # Create a new solution
            new_solution = self._nests[i] + self._random.rand() * (self._nests[j] - self._nests[k])
            new_solution = np.clip(new_solution, self.lower_bound, self.upper_bound)
            
            # Replace the abandoned nest
            new_fitness = self.benchmark_func.evaluate(new_solution)
            self._nests[i] = new_solution
            self._fitness[i] = new_fitness
