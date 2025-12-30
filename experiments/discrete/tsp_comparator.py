import numpy as np
import sys
import os
from typing import Dict, List, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.algorithms.discrete.tsp_problem import TSPProblem, create_tsp
from src.algorithms.discrete.aco import ACO
from src.algorithms.discrete.pso import PSO
from src.algorithms.discrete.genetic_algorithm import GeneticAlgorithm
from algorithm_result import AlgorithmResult


class TSPComparator:
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.results: List[AlgorithmResult] = []
        
    def run_aco(
        self,
        tsp: TSPProblem,
        n_ants: int = 30,
        n_iterations: int = 100,
        alpha: float = 2.0,
        beta: float = 5.0,
        rho: float = 0.5,
        Q: float = 100.0
    ) -> AlgorithmResult:
        print(f"\nRunning ACO (ants={n_ants}, iterations={n_iterations})...")
        
        aco = ACO(
            tsp_problem=tsp,
            n_ants=n_ants,
            n_iterations=n_iterations,
            alpha=alpha,
            beta=beta,
            rho=rho,
            Q=Q
        )
        
        best_tour, best_cost, cost_history, elapsed_time = aco.solve()
        
        result = AlgorithmResult(
            algorithm_name="ACO",
            best_tour=best_tour,
            best_cost=best_cost,
            cost_history=cost_history,
            elapsed_time=elapsed_time,
            iterations=len(cost_history),
            parameters={
                'n_ants': n_ants,
                'alpha': alpha,
                'beta': beta,
                'rho': rho,
                'Q': Q
            }
        )
        
        return result
    
    def run_pso(
        self,
        tsp: TSPProblem,
        n_particles: int = 30,
        n_iterations: int = 100,
        w: float = 0.7,
        c1: float = 0.9,
        c2: float = 0.3,
        topology: str = 'gbest',
        neighborhood_size: int = 5
    ) -> AlgorithmResult:
        print(f"\nRunning PSO-{topology} (particles={n_particles}, iterations={n_iterations})...")
        
        pso = PSO(
            tsp_problem=tsp,
            n_particles=n_particles,
            n_iterations=n_iterations,
            w=w,
            c1=c1,
            c2=c2,
            topology=topology,
            neighborhood_size=neighborhood_size,
            seed=self.seed
        )
        
        best_tour, best_cost, cost_history, elapsed_time = pso.solve()
        
        algo_name = f"PSO-{topology}"
        if topology == 'lbest':
            algo_name += f"(k={neighborhood_size})"
        
        result = AlgorithmResult(
            algorithm_name=algo_name,
            best_tour=best_tour,
            best_cost=best_cost,
            cost_history=cost_history,
            elapsed_time=elapsed_time,
            iterations=len(cost_history),
            parameters={
                'n_particles': n_particles,
                'w': w,
                'c1': c1,
                'c2': c2,
                'topology': topology,
                'neighborhood_size': neighborhood_size if topology == 'lbest' else None
            }
        )
        
        return result
    
    def run_ga(
        self,
        tsp: TSPProblem,
        population_size: int = 100,
        n_generations: int = 500,
        p_crossover: float = 0.8,
        p_mutation: float = 0.2,
        tournament_k: int = 5
    ) -> AlgorithmResult:
        print(f"\nRunning GA (population={population_size}, generations={n_generations})...")
        
        ga = GeneticAlgorithm(
            tsp_problem=tsp,
            population_size=population_size,
            n_generations=n_generations,
            p_crossover=p_crossover,
            p_mutation=p_mutation,
            tournament_k=tournament_k,
            seed=self.seed
        )
        
        best_tour, best_cost, cost_history, fitness_history, elapsed_time = ga.solve()
        
        result = AlgorithmResult(
            algorithm_name="GA",
            best_tour=best_tour,
            best_cost=best_cost,
            cost_history=cost_history,
            elapsed_time=elapsed_time,
            iterations=len(cost_history),
            parameters={
                'population_size': population_size,
                'p_crossover': p_crossover,
                'p_mutation': p_mutation,
                'tournament_k': tournament_k
            }
        )
        
        return result
    
    def compare_algorithms(
        self,
        tsp: TSPProblem,
        algorithms: List[str] = None
    ) -> List[AlgorithmResult]:
        if algorithms is None:
            algorithms = ['ACO', 'PSO-gbest', 'PSO-lbest', 'GA']
        
        results = []
        n_cities = tsp.n_cities
        
        n_iterations = min(200, max(50, n_cities * 2))
        
        for algo in algorithms:
            if algo == 'ACO':
                result = self.run_aco(tsp, n_ants=50, n_iterations=n_iterations)
            elif algo == 'PSO-gbest':
                result = self.run_pso(tsp, n_particles=50, n_iterations=n_iterations, topology='gbest')
            elif algo == 'PSO-lbest':
                result = self.run_pso(tsp, n_particles=50, n_iterations=n_iterations, 
                                     topology='lbest', neighborhood_size=5)
            elif algo == 'GA':
                result = self.run_ga(tsp, population_size=50, n_generations=n_iterations)
            else:
                print(f"Unknown algorithm: {algo}")
                continue
            
            results.append(result)
        
        self.results = results
        return results
    
    def robustness_analysis(
        self,
        n_cities: int = 50,
        n_runs: int = 10,
        algorithms: List[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        print("\n" + "="*70)
        print(f"ROBUSTNESS ANALYSIS ({n_runs} runs, {n_cities} cities)")
        print("="*70)
        
        if algorithms is None:
            algorithms = ['ACO', 'PSO-gbest', 'PSO-lbest', 'GA']
        
        stats = {}
        
        for run in range(n_runs):
            print(f"\n--- Run {run + 1}/{n_runs} ---")
            
            tsp = create_tsp(n_cities=n_cities, symmetric=True, seed=self.seed + run)
            
            results = self.compare_algorithms(tsp, algorithms)
            
            for result in results:
                algo_name = result.algorithm_name
                if algo_name not in stats:
                    stats[algo_name] = {'costs': [], 'times': []}
                stats[algo_name]['costs'].append(result.best_cost)
                stats[algo_name]['times'].append(result.elapsed_time)
        
        summary = {}
        for algo_name in stats.keys():
            costs = stats[algo_name]['costs']
            times = stats[algo_name]['times']
            
            summary[algo_name] = {
                'mean_cost': np.mean(costs),
                'std_cost': np.std(costs),
                'min_cost': np.min(costs),
                'max_cost': np.max(costs),
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'coefficient_of_variation': np.std(costs) / np.mean(costs) if np.mean(costs) > 0 else 0
            }
        
        return summary
    
    def scalability_analysis(
        self,
        city_sizes: List[int] = None,
        algorithms: List[str] = None
    ) -> Dict[str, Dict[str, List[float]]]:
        if city_sizes is None:
            city_sizes = [10, 20, 30, 50, 75, 100]
        
        if algorithms is None:
            algorithms = ['ACO', 'PSO-gbest', 'PSO-lbest', 'GA']
        
        print("\n" + "="*70)
        print(f"SCALABILITY ANALYSIS")
        print("="*70)
        
        scalability_data = {}
        
        for n_cities in city_sizes:
            print(f"\n--- Testing with {n_cities} cities ---")
            
            tsp = create_tsp(n_cities=n_cities, symmetric=True, seed=self.seed)
            results = self.compare_algorithms(tsp, algorithms)
            
            for result in results:
                algo_name = result.algorithm_name
                if algo_name not in scalability_data:
                    scalability_data[algo_name] = {'sizes': [], 'costs': [], 'times': [], 'iterations': []}
                
                scalability_data[algo_name]['sizes'].append(n_cities)
                scalability_data[algo_name]['costs'].append(result.best_cost)
                scalability_data[algo_name]['times'].append(result.elapsed_time)
                scalability_data[algo_name]['iterations'].append(result.iterations)
        
        return scalability_data
    
    def parameter_sensitivity_aco(
        self,
        tsp: TSPProblem,
        param_ranges: Dict[str, List[float]] = None
    ) -> Dict[str, Dict[str, List[float]]]:
        if param_ranges is None:
            param_ranges = {
                'alpha': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
                'beta': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                'rho': [0.1, 0.3, 0.5, 0.7, 0.9],
                'n_ants': [10, 20, 30, 50, 75, 100]
            }
        
        print("\n" + "="*70)
        print("ACO PARAMETER SENSITIVITY ANALYSIS")
        print("="*70)
        
        sensitivity_data = {}
        
        print("\nTesting alpha (pheromone weight)...")
        sensitivity_data['alpha'] = {'values': [], 'costs': [], 'times': []}
        for alpha in param_ranges['alpha']:
            result = self.run_aco(tsp, alpha=alpha, beta=5.0, n_iterations=100)
            sensitivity_data['alpha']['values'].append(alpha)
            sensitivity_data['alpha']['costs'].append(result.best_cost)
            sensitivity_data['alpha']['times'].append(result.elapsed_time)
        
        print("\nTesting beta (heuristic weight)...")
        sensitivity_data['beta'] = {'values': [], 'costs': [], 'times': []}
        for beta in param_ranges['beta']:
            result = self.run_aco(tsp, alpha=2.0, beta=beta, n_iterations=100)
            sensitivity_data['beta']['values'].append(beta)
            sensitivity_data['beta']['costs'].append(result.best_cost)
            sensitivity_data['beta']['times'].append(result.elapsed_time)
        
        print("\nTesting rho (evaporation rate)...")
        sensitivity_data['rho'] = {'values': [], 'costs': [], 'times': []}
        for rho in param_ranges['rho']:
            result = self.run_aco(tsp, rho=rho, n_iterations=100)
            sensitivity_data['rho']['values'].append(rho)
            sensitivity_data['rho']['costs'].append(result.best_cost)
            sensitivity_data['rho']['times'].append(result.elapsed_time)
        
        print("\nTesting n_ants (colony size)...")
        sensitivity_data['n_ants'] = {'values': [], 'costs': [], 'times': []}
        for n_ants in param_ranges['n_ants']:
            result = self.run_aco(tsp, n_ants=n_ants, n_iterations=100)
            sensitivity_data['n_ants']['values'].append(n_ants)
            sensitivity_data['n_ants']['costs'].append(result.best_cost)
            sensitivity_data['n_ants']['times'].append(result.elapsed_time)
        
        return sensitivity_data
    
    def parameter_sensitivity_pso(
        self,
        tsp: TSPProblem,
        param_ranges: Dict[str, List[float]] = None
    ) -> Dict[str, Dict[str, List[float]]]:
        if param_ranges is None:
            param_ranges = {
                'w': [0.1, 0.3, 0.5, 0.7, 0.9],
                'c1': [0.1, 0.3, 0.5, 0.7, 0.9],
                'c2': [0.1, 0.3, 0.5, 0.7, 0.9],
                'n_particles': [10, 20, 30, 50, 75, 100]
            }
        
        print("\n" + "="*70)
        print("PSO PARAMETER SENSITIVITY ANALYSIS")
        print("="*70)
        
        sensitivity_data = {}
        
        print("\nTesting w (inertia weight)...")
        sensitivity_data['w'] = {'values': [], 'costs': [], 'times': []}
        for w in param_ranges['w']:
            result = self.run_pso(tsp, w=w, c1=0.3, c2=0.7, n_iterations=100)
            sensitivity_data['w']['values'].append(w)
            sensitivity_data['w']['costs'].append(result.best_cost)
            sensitivity_data['w']['times'].append(result.elapsed_time)
        
        print("\nTesting c1 (cognitive coefficient)...")
        sensitivity_data['c1'] = {'values': [], 'costs': [], 'times': []}
        for c1 in param_ranges['c1']:
            result = self.run_pso(tsp, w=0.7, c1=c1, c2=0.7, n_iterations=100)
            sensitivity_data['c1']['values'].append(c1)
            sensitivity_data['c1']['costs'].append(result.best_cost)
            sensitivity_data['c1']['times'].append(result.elapsed_time)
        
        print("\nTesting c2 (social coefficient)...")
        sensitivity_data['c2'] = {'values': [], 'costs': [], 'times': []}
        for c2 in param_ranges['c2']:
            result = self.run_pso(tsp, w=0.7, c1=0.3, c2=c2, n_iterations=100)
            sensitivity_data['c2']['values'].append(c2)
            sensitivity_data['c2']['costs'].append(result.best_cost)
            sensitivity_data['c2']['times'].append(result.elapsed_time)
        
        print("\nTesting n_particles (swarm size)...")
        sensitivity_data['n_particles'] = {'values': [], 'costs': [], 'times': []}
        for n_particles in param_ranges['n_particles']:
            result = self.run_pso(tsp, w=0.7, c1=0.3, c2=0.7, n_particles=n_particles, n_iterations=100)
            sensitivity_data['n_particles']['values'].append(n_particles)
            sensitivity_data['n_particles']['costs'].append(result.best_cost)
            sensitivity_data['n_particles']['times'].append(result.elapsed_time)
        
        return sensitivity_data
    
    def parameter_sensitivity_ga(
        self,
        tsp: TSPProblem,
        param_ranges: Dict[str, List[float]] = None
    ) -> Dict[str, Dict[str, List[float]]]:
        if param_ranges is None:
            param_ranges = {
                'p_crossover': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                'p_mutation': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
                'tournament_k': [2, 3, 5, 7, 10, 15],
                'population_size': [20, 50, 100, 150, 200]
            }
        
        print("\n" + "="*70)
        print("GA PARAMETER SENSITIVITY ANALYSIS")
        print("="*70)
        
        sensitivity_data = {}
        
        print("\nTesting p_crossover (crossover probability)...")
        sensitivity_data['p_crossover'] = {'values': [], 'costs': [], 'times': []}
        for p_crossover in param_ranges['p_crossover']:
            result = self.run_ga(tsp, p_crossover=p_crossover, n_generations=200)
            sensitivity_data['p_crossover']['values'].append(p_crossover)
            sensitivity_data['p_crossover']['costs'].append(result.best_cost)
            sensitivity_data['p_crossover']['times'].append(result.elapsed_time)
        
        print("\nTesting p_mutation (mutation probability)...")
        sensitivity_data['p_mutation'] = {'values': [], 'costs': [], 'times': []}
        for p_mutation in param_ranges['p_mutation']:
            result = self.run_ga(tsp, p_mutation=p_mutation, n_generations=200)
            sensitivity_data['p_mutation']['values'].append(p_mutation)
            sensitivity_data['p_mutation']['costs'].append(result.best_cost)
            sensitivity_data['p_mutation']['times'].append(result.elapsed_time)
        
        print("\nTesting tournament_k (tournament size)...")
        sensitivity_data['tournament_k'] = {'values': [], 'costs': [], 'times': []}
        for tournament_k in param_ranges['tournament_k']:
            result = self.run_ga(tsp, tournament_k=tournament_k, n_generations=200)
            sensitivity_data['tournament_k']['values'].append(tournament_k)
            sensitivity_data['tournament_k']['costs'].append(result.best_cost)
            sensitivity_data['tournament_k']['times'].append(result.elapsed_time)
        
        print("\nTesting population_size...")
        sensitivity_data['population_size'] = {'values': [], 'costs': [], 'times': []}
        for population_size in param_ranges['population_size']:
            result = self.run_ga(tsp, population_size=population_size, n_generations=200)
            sensitivity_data['population_size']['values'].append(population_size)
            sensitivity_data['population_size']['costs'].append(result.best_cost)
            sensitivity_data['population_size']['times'].append(result.elapsed_time)
        
        return sensitivity_data
