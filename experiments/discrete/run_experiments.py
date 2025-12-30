"""
Comprehensive Comparison of Discrete Optimization Algorithms for TSP

This module compares:
- Ant Colony Optimization (ACO)
- Particle Swarm Optimization (PSO) - gbest and lbest
- Genetic Algorithm (GA)

Comparison metrics:
1. Convergence speed
2. Computational complexity (time and space)
3. Robustness (performance across multiple runs)
4. Scalability (performance with problem size)
5. Parameter sensitivity analysis

"""

from demo_functions import (
    demo_basic_comparison,
    demo_robustness_analysis,
    demo_scalability_analysis,
    demo_parameter_sensitivity,
    demo_comprehensive_analysis
)


if __name__ == "__main__":
    demo_basic_comparison()
    # demo_parameter_sensitivity()
    # demo_robustness_analysis()
    # demo_scalability_analysis()
    # demo_comprehensive_analysis()

