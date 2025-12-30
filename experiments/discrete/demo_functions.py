import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.algorithms.discrete.tsp_problem import create_tsp
from tsp_comparator import TSPComparator
from visualizer import Visualizer


def demo_basic_comparison():
    print("\n" + "="*70)
    print("DEMO 1: BASIC ALGORITHM COMPARISON")
    print("="*70)
    
    N_CITIES = 50
    tsp = create_tsp(n_cities=N_CITIES, symmetric=True, seed=42)
    
    comparator = TSPComparator(seed=42)
    results = comparator.compare_algorithms(tsp)
    
    visualizer = Visualizer()
    visualizer.create_summary_table(results)
    visualizer.plot_convergence_comparison(results, 
                                          title=f"Convergence Comparison ({N_CITIES} cities)")
    visualizer.plot_performance_comparison(results)


def demo_robustness_analysis():
    print("\n" + "="*70)
    print("DEMO 2: ROBUSTNESS ANALYSIS")
    print("="*70)
    
    comparator = TSPComparator(seed=42)
    robustness_stats = comparator.robustness_analysis(
        n_cities=50,
        n_runs=10,
        algorithms=['ACO', 'PSO-gbest', 'PSO-lbest', 'GA']
    )
    
    print("\n" + "="*70)
    print("ROBUSTNESS STATISTICS")
    print("="*70)
    for algo, stats in robustness_stats.items():
        print(f"\n{algo}:")
        print(f"  Mean Cost: {stats['mean_cost']:.2f} ± {stats['std_cost']:.2f}")
        print(f"  Min/Max: {stats['min_cost']:.2f} / {stats['max_cost']:.2f}")
        print(f"  Coefficient of Variation: {stats['coefficient_of_variation']:.4f}")
        print(f"  Mean Time: {stats['mean_time']:.2f}s ± {stats['std_time']:.2f}s")
    
    visualizer = Visualizer()
    visualizer.plot_robustness_analysis(robustness_stats)


def demo_scalability_analysis():
    print("\n" + "="*70)
    print("DEMO 3: SCALABILITY ANALYSIS")
    print("="*70)
    
    comparator = TSPComparator(seed=42)
    scalability_data = comparator.scalability_analysis(
        city_sizes=[10, 20, 30, 50, 100],
        algorithms=['ACO', 'PSO-gbest', 'PSO-lbest', 'GA']
    )
    
    visualizer = Visualizer()
    visualizer.plot_scalability_analysis(scalability_data)


def demo_parameter_sensitivity():
    print("\n" + "="*70)
    print("DEMO 4: PARAMETER SENSITIVITY ANALYSIS")
    print("="*70)
    
    N_CITIES = 40
    tsp = create_tsp(n_cities=N_CITIES, symmetric=True, seed=42)
    
    comparator = TSPComparator(seed=42)
    visualizer = Visualizer()
    
    print("\n--- PSO Parameter Sensitivity ---")
    pso_sensitivity = comparator.parameter_sensitivity_pso(tsp)
    visualizer.plot_parameter_sensitivity(pso_sensitivity, "PSO")
    
    print("\n--- GA Parameter Sensitivity ---")
    ga_sensitivity = comparator.parameter_sensitivity_ga(tsp)
    visualizer.plot_parameter_sensitivity(ga_sensitivity, "GA")


def demo_comprehensive_analysis():
    print("\n" + "="*70)
    print("COMPREHENSIVE ALGORITHM ANALYSIS")
    print("="*70)
    
    print("\n[1/4] Running basic comparison...")
    N_CITIES = 40
    tsp = create_tsp(n_cities=N_CITIES, symmetric=True, seed=42)
    
    comparator = TSPComparator(seed=42)
    results = comparator.compare_algorithms(tsp)
    
    visualizer = Visualizer()
    visualizer.create_summary_table(results)
    visualizer.plot_convergence_comparison(results, 
                                          title=f"Convergence Comparison ({N_CITIES} cities)")
    visualizer.plot_performance_comparison(results)
    
    print("\n[2/4] Running robustness analysis...")
    robustness_stats = comparator.robustness_analysis(
        n_cities=30,
        n_runs=3,
        algorithms=['ACO', 'PSO-gbest', 'PSO-lbest', 'GA']
    )
    visualizer.plot_robustness_analysis(robustness_stats)
    
    print("\n[3/4] Running scalability analysis...")
    scalability_data = comparator.scalability_analysis(
        city_sizes=[10, 20, 30, 40],
        algorithms=['ACO', 'PSO-gbest', 'PSO-lbest', 'GA']
    )
    visualizer.plot_scalability_analysis(scalability_data)
    
    print("\n[4/4] Running parameter sensitivity analysis...")
    tsp_small = create_tsp(n_cities=25, symmetric=True, seed=42)
    aco_sensitivity = comparator.parameter_sensitivity_aco(tsp_small)
    visualizer.plot_parameter_sensitivity(aco_sensitivity, "ACO")
    
    print("\n" + "="*70)
    print("COMPREHENSIVE ANALYSIS COMPLETED")
    print("="*70)
