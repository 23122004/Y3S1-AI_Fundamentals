import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any
from algorithm_result import AlgorithmResult


class Visualizer:
    
    @staticmethod
    def plot_convergence_comparison(
        results: List[AlgorithmResult],
        title: str = "Algorithm Convergence Comparison",
        save_path: str = None
    ):
        plt.figure(figsize=(12, 7))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for i, result in enumerate(results):
            color = colors[i % len(colors)]
            plt.plot(result.cost_history, linewidth=2, label=result.algorithm_name, color=color)
        
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Best Cost', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_performance_comparison(
        results: List[AlgorithmResult],
        save_path: str = None
    ):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        algo_names = [r.algorithm_name for r in results]
        best_costs = [r.best_cost for r in results]
        elapsed_times = [r.elapsed_time for r in results]
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        axes[0].bar(algo_names, best_costs, color=colors[:len(algo_names)])
        axes[0].set_ylabel('Best Cost', fontsize=11)
        axes[0].set_title('Solution Quality', fontsize=12, fontweight='bold')
        axes[0].tick_params(axis='x', rotation=15)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        axes[1].bar(algo_names, elapsed_times, color=colors[:len(algo_names)])
        axes[1].set_ylabel('Time (seconds)', fontsize=11)
        axes[1].set_title('Computational Time', fontsize=12, fontweight='bold')
        axes[1].tick_params(axis='x', rotation=15)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_robustness_analysis(
        robustness_stats: Dict[str, Dict[str, Any]],
        save_path: str = None
    ):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        algo_names = list(robustness_stats.keys())
        mean_costs = [robustness_stats[algo]['mean_cost'] for algo in algo_names]
        std_costs = [robustness_stats[algo]['std_cost'] for algo in algo_names]
        cv = [robustness_stats[algo]['coefficient_of_variation'] for algo in algo_names]
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        axes[0].bar(algo_names, mean_costs, yerr=std_costs, capsize=5, 
                   color=colors[:len(algo_names)], alpha=0.7)
        axes[0].set_ylabel('Mean Best Cost ± Std Dev', fontsize=11)
        axes[0].set_title('Robustness: Cost Statistics', fontsize=12, fontweight='bold')
        axes[0].tick_params(axis='x', rotation=15)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        axes[1].bar(algo_names, cv, color=colors[:len(algo_names)], alpha=0.7)
        axes[1].set_ylabel('Coefficient of Variation (CV)', fontsize=11)
        axes[1].set_title('Robustness: Solution Stability', fontsize=12, fontweight='bold')
        axes[1].tick_params(axis='x', rotation=15)
        axes[1].grid(True, alpha=0.3, axis='y')
        axes[1].axhline(y=0.05, color='r', linestyle='--', linewidth=1, label='Good (CV<0.05)')
        axes[1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_scalability_analysis(
        scalability_data: Dict[str, Dict[str, List[float]]],
        save_path: str = None
    ):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for i, (algo, data) in enumerate(scalability_data.items()):
            color = colors[i % len(colors)]
            axes[0].plot(data['sizes'], data['times'], marker='o', linewidth=2,
                        label=algo, color=color)
        
        axes[0].set_xlabel('Number of Cities', fontsize=11)
        axes[0].set_ylabel('Execution Time (seconds)', fontsize=11)
        axes[0].set_title('Scalability: Time Complexity', fontsize=12, fontweight='bold')
        axes[0].legend(loc='upper left', fontsize=9)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale('log')
        
        for i, (algo, data) in enumerate(scalability_data.items()):
            color = colors[i % len(colors)]
            axes[1].plot(data['sizes'], data['costs'], marker='s', linewidth=2,
                        label=algo, color=color)
        
        axes[1].set_xlabel('Number of Cities', fontsize=11)
        axes[1].set_ylabel('Best Cost Found', fontsize=11)
        axes[1].set_title('Scalability: Solution Quality', fontsize=12, fontweight='bold')
        axes[1].legend(loc='upper left', fontsize=9)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_parameter_sensitivity(
        sensitivity_data: Dict[str, Dict[str, List[float]]],
        algorithm_name: str,
        save_path: str = None
    ):
        n_params = len(sensitivity_data)
        fig, axes = plt.subplots(2, (n_params + 1) // 2, figsize=(15, 8))
        axes = axes.flatten()
        
        for idx, (param, data) in enumerate(sensitivity_data.items()):
            ax = axes[idx]
            ax.plot(data['values'], data['costs'], marker='o', linewidth=2, color='#1f77b4')
            ax.set_xlabel(param, fontsize=10)
            ax.set_ylabel('Best Cost', fontsize=10)
            ax.set_title(f'{param.upper()} Sensitivity', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            best_idx = np.argmin(data['costs'])
            best_value = data['values'][best_idx]
            best_cost = data['costs'][best_idx]
            ax.scatter([best_value], [best_cost], color='red', s=100, zorder=5,
                      label=f'Best: {best_value}')
            ax.legend(fontsize=8)
        
        for idx in range(n_params, len(axes)):
            axes[idx].axis('off')
        
        fig.suptitle(f'{algorithm_name} Parameter Sensitivity Analysis', 
                    fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def create_summary_table(
        results: List[AlgorithmResult],
        save_path: str = None
    ):
        print("\n" + "="*90)
        print("ALGORITHM COMPARISON SUMMARY")
        print("="*90)
        print(f"{'Algorithm':<20} {'Best Cost':>15} {'Time (s)':>12} {'Iterations':>12} {'Avg Cost/Iter':>15}")
        print("-"*90)
        
        for result in results:
            avg_cost_per_iter = result.best_cost / result.iterations if result.iterations > 0 else 0
            print(f"{result.algorithm_name:<20} {result.best_cost:>15.2f} {result.elapsed_time:>12.2f} "
                  f"{result.iterations:>12} {avg_cost_per_iter:>15.2f}")
        
        print("="*90)
        
        best_cost_algo = min(results, key=lambda x: x.best_cost)
        fastest_algo = min(results, key=lambda x: x.elapsed_time)
        
        print(f"\n🏆 Best Solution Quality: {best_cost_algo.algorithm_name} (Cost: {best_cost_algo.best_cost:.2f})")
        print(f"⚡ Fastest Algorithm: {fastest_algo.algorithm_name} (Time: {fastest_algo.elapsed_time:.2f}s)")
        print("="*90 + "\n")
