from .base import Optimizer
import numpy as np

class ArtificialBeeColony(Optimizer):
    def __init__(self, benchmark_func, max_iter, random_state=None, population_size=50, limit=10):
        super().__init__(benchmark_func, max_iter, random_state)
        self.population_size = population_size
        self.limit = limit
        
        # Number of food sources = Number of employed bees
        self.n_sources = population_size // 2
        # Number of onlooker bees
        self.n_onlooker = population_size - self.n_sources

       # Initialize population
        self._food_sources = self._random.uniform(self.lower_bound, self.upper_bound, (self.n_sources, self.dim))
        self._fitness = np.array([self.benchmark_func.evaluate(source) for source in self._food_sources])
        self._trial_counters = np.zeros(self.n_sources)

        # Initial best and solution
        best_source_id = np.argmin(self._fitness)
        self._best_fitness = self._fitness[best_source_id]
        self._best_solution = self._food_sources[best_source_id].copy()
        
        # Convergence history
        self._convergence_history.append(self._best_fitness)
        self._solution_history.append(self._best_solution)
        self._population_history.append(self._food_sources.copy())

    def solve(self):
        """
        Main loop of Artificial Bee Colony algorithm
        """
        for _ in range(self.max_iter):
            # Employed bees search for new food sources and update their memory
            self._employed_bee_phase()
            
            # Calculate selection probabilities for onlooker bees
            probabilities = self._calculate_probabilities()
            
            # Onlooker bees select food sources based on probability and search
            self._onlooker_bee_phase(probabilities)
            
            # Scout bees check for abandoned food sources and replace them
            self._scout_bee_phase()

            # Update the current best
            curr_best_id = np.argmin(self._fitness)
            if self._fitness[curr_best_id] < self._best_fitness:
                self._best_fitness = self._fitness[curr_best_id]
                self._best_solution = self._food_sources[curr_best_id].copy()

            self._convergence_history.append(self._best_fitness)
            self._solution_history.append(self._best_solution)
            self._population_history.append(self._food_sources.copy())

        return self._best_solution, self._best_fitness
    
    def _get_neighbor(self, i):
        k = i
        while k == i:
            k = self._random.randint(0, self.n_sources)

        # Select a random dimension to modify
        j = self._random.randint(0, self.dim)
        phi = self._random.uniform(-1, 1)
        
        #Create the new solution
        neighbor = self._food_sources[i].copy()
        neighbor[j] = self._food_sources[i][j] + phi * (self._food_sources[i][j] - self._food_sources[k][j])
        
        return np.clip(neighbor, self.lower_bound, self.upper_bound)

    def _employed_bee_phase(self):
        for i in range(self.n_sources):
            neighbor = self._get_neighbor(i)
            neighbor_fit = self.benchmark_func.evaluate(neighbor)
            
            # Replace if better
            if neighbor_fit < self._fitness[i]:
                self._food_sources[i] = neighbor
                self._fitness[i] = neighbor_fit
                self._trial_counters[i] = 0
            else:
                self._trial_counters[i] += 1

    def _calculate_probabilities(self):
        fitness_values = np.zeros_like(self._fitness)
        
        # Handle non-negative fitness values
        non_neg_idx = self._fitness >= 0
        fitness_values[non_neg_idx] = 1.0 / (1.0 + self._fitness[non_neg_idx])
        
        # Handle negative fitness values
        neg_idx = self._fitness < 0
        fitness_values[neg_idx] = 1.0 + np.abs(self._fitness[neg_idx])
        
        # Sum of all nectar amounts
        fit_sum = np.sum(fitness_values)
        
        # Avoid division by zero
        if fit_sum == 0:
            return np.ones(self.n_sources) / self.n_sources
        else:
            return fitness_values / fit_sum

    def _onlooker_bee_phase(self, probabilities):
        for _ in range(self.n_onlooker):
            # Select a food source 'j' using Roulette Wheel Selection
            j = self._random.choice(range(self.n_sources), p=probabilities)
            
            neighbor = self._get_neighbor(j)
            neighbor_fit = self.benchmark_func.evaluate(neighbor)
            
            # Replace if better
            if neighbor_fit < self._fitness[j]:
                self._food_sources[j] = neighbor
                self._fitness[j] = neighbor_fit
                self._trial_counters[j] = 0
            else:
                self._trial_counters[j] += 1
    
    def _scout_bee_phase(self):
        for i in range(self.n_sources):
            # If trial counter exceeds limit, replace food source
            if self._trial_counters[i] > self.limit:
                self._food_sources[i] = self.lower_bound + (self.upper_bound - self.lower_bound) * self._random.rand(self.dim)
                self._fitness[i] = self.benchmark_func.evaluate(self._food_sources[i])
                self._trial_counters[i] = 0
