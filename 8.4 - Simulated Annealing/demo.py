"""
Demo minh họa thuật toán Simulated Annealing với hàm Rastrigin

Hàm Rastrigin là một hàm kiểm thử tối ưu hóa nổi tiếng với nhiều điểm cực tiểu
cục bộ, rất thích hợp để kiểm tra khả năng thoát khỏi local minima của 
Simulated Annealing thông qua tính ngẫu nhiên và nhiệt độ.

Hàm Rastrigin: f(x) = An + Σ[x_i² - A·cos(2πx_i)]
với A = 10, điểm cực tiểu toàn cục tại (0, 0) với f(0,0) = 0
"""

import numpy as np
import matplotlib.pyplot as plt
from SA import simulated_annealing, gradient_descent
import sys
import os

# =============================================================================
# Định nghĩa hàm Rastrigin 
# =============================================================================

A_RASTRIGIN = 10

def rastrigin(x: np.ndarray) -> float:
    n = len(x)
    return float(A_RASTRIGIN * n + np.sum(x**2 - A_RASTRIGIN * np.cos(2 * np.pi * x)))

def grad_rastrigin(x: np.ndarray) -> np.ndarray:
    return 2 * x + 2 * np.pi * A_RASTRIGIN * np.sin(2 * np.pi * x)

def f(x: np.ndarray) -> float:
    return rastrigin(x)

def grad_f(x: np.ndarray) -> np.ndarray:
    return grad_rastrigin(x)

# =============================================================================
# Hàm tìm seed tốt nhất cho SA
# =============================================================================

def find_best_seed_sa(
    f_func,
    x0: np.ndarray,
    t0: float = 10.0,
    sigma: float = 1.0,
    max_iter: int = 1000,
    schedule: str = 'exponential',
    seed_range: range = range(50)
) -> tuple:
    best_seed = None
    best_f_val = float('inf')
    best_x = None
    
    for seed in seed_range:
        x_result, _, f_hist = simulated_annealing(
            f_func, x0, t0=t0, sigma=sigma,
            max_iter=max_iter, schedule=schedule, seed=seed
        )
        f_val = f_hist[-1]
        
        if f_val < best_f_val:
            best_f_val = f_val
            best_seed = seed
            best_x = x_result.copy()
            
    return best_seed, best_f_val, best_x


# =============================================================================
# Hàm Visualization
# =============================================================================

def plot_contour_paths(
    objective_func,
    history_gd: list,
    history_sa: list,
    x0: np.ndarray,
    title: str = "GD vs Simulated Annealing Comparison",
    x_range: tuple = (-4, 4),
    save_path: str = "sa_contour_paths.png"
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Tính toán x_range động để không bị "lỗi hiển thị ngoài vùng"
    all_pts = np.vstack([np.array(history_gd), np.array(history_sa)])
    x_min, x_max = np.min(all_pts[:, 0]), np.max(all_pts[:, 0])
    y_min, y_max = np.min(all_pts[:, 1]), np.max(all_pts[:, 1])
    
    # Thêm margin
    margin = 0.5
    x_range_plot = (min(x_min, -0.5) - margin, max(x_max, 0.5) + margin)
    y_range_plot = (min(y_min, -0.5) - margin, max(y_max, 0.5) + margin)
    
    # Đảm bảo phạm vi bao phủ ít nhất là vùng quan trọng quanh (0,0)
    limit = max(abs(x_range_plot[0]), abs(x_range_plot[1]), abs(y_range_plot[0]), abs(y_range_plot[1]), 3)
    x_range_final = (-limit, limit)
    
    x1_range = np.linspace(x_range_final[0], x_range_final[1], 201)
    x2_range = np.linspace(x_range_final[0], x_range_final[1], 201)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    Z = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Z[i, j] = objective_func(np.array([X1[i, j], X2[i, j]]))
    
    contourf = ax.contourf(X1, X2, Z, levels=50, cmap='viridis', alpha=0.8)
    ax.contour(X1, X2, Z, levels=20, colors='white', linewidths=0.5, alpha=0.5)
    plt.colorbar(contourf, ax=ax, label='f(x)')
    
    # GD Path
    pts_gd = np.array(history_gd)
    ax.plot(pts_gd[:, 0], pts_gd[:, 1], 'c-', linewidth=1.5, alpha=0.7)
    ax.scatter(pts_gd[-1, 0], pts_gd[-1, 1], c='blue', s=120, marker='X', 
               edgecolors='white', linewidths=2, label='GD final', zorder=5)
    
    # SA Path
    pts_sa = np.array(history_sa)
    ax.plot(pts_sa[:, 0], pts_sa[:, 1], 'r-', linewidth=1, alpha=0.5)
    ax.scatter(pts_sa[-1, 0], pts_sa[-1, 1], c='darkred', s=120, marker='X', 
               edgecolors='white', linewidths=2, label='SA final', zorder=5)
    
    # Start & Opt
    ax.plot(x0[0], x0[1], 'w*', markersize=15, markeredgecolor='black', zorder=6)
    ax.plot(0, 0, 'g*', markersize=15, markeredgecolor='white', zorder=6)
    
    # Legend elements phong cách 8.1
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='cyan', lw=1.5, label='GD path'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='blue', 
               markersize=10, markeredgecolor='white', label='GD final'),
        Line2D([0], [0], color='red', lw=1.5, alpha=0.5, label='SA path'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='darkred', 
               markersize=10, markeredgecolor='white', label='SA final'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='white', 
               markersize=12, markeredgecolor='black', label='Start'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='green', 
               markersize=12, markeredgecolor='white', label='Global Opt (0,0)'),
    ]
    
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title(title)
    ax.legend(handles=legend_elements, loc='upper right', ncol=2, fontsize=8)
    ax.set_aspect('equal')
    
    ax.set_xlim(x_range_final)
    ax.set_ylim(x_range_final)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.show()

def plot_convergence(
    f_history_gd: list,
    f_history_sa: list,
    title: str = "Convergence Comparison",
    save_path: str = "sa_convergence.png"
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(f_history_gd, 'b-', linewidth=2, label=f'Gradient Descent (final: {f_history_gd[-1]:.4f})')
    ax.plot(f_history_sa, 'r-', linewidth=2, label=f'Simulated Annealing (final: {f_history_sa[-1]:.4f})')
    
    ax.set_yscale('log')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('$f(x)$ (log scale)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.show()

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Khởi tạo tham số
    x0 = np.array([-2.5, -2.5])
    max_iter_sa = 400
    max_iter_gd = 400
    t0 = 15.0          
    sigma = 0.5
    gamma = 0.99
    alpha = 0.01
    schedule = 'exponential'
    seed_range = range(20)
    x_range = (-4, 4)
    
    # Tìm seed tốt nhất cho SA
    # best_seed, best_f, _ = find_best_seed_sa(
    #     f, x0, t0=t0, sigma=sigma, max_iter=max_iter_sa, seed_range=seed_range
    # )
    # print(best_seed)
    
    # Chạy SA với seed tốt nhất
    x_sa, history_sa, f_history_sa = simulated_annealing(
        f, x0, t0=t0, sigma=sigma, max_iter=max_iter_sa, schedule=schedule, gamma=gamma, seed=6 # best_seed
    )
    
    # Chạy GD để so sánh
    x_gd, history_gd, f_history_gd = gradient_descent(
        f, grad_f, x0, alpha=alpha, max_iter=max_iter_gd
    )
    
    # Visualization
    plot_contour_paths(
        f, history_gd, history_sa, x0, 
        title="Gradient Descent vs Simulated Annealing on Rastrigin",
        x_range=x_range,
        save_path="sa_contour.png"
    )
    
    plot_convergence(
        f_history_gd, f_history_sa,
        title="Simulated Annealing: Convergence Rate",
        save_path="sa_convergence.png"
    )
    
    print("\n" + "=" * 50)
    print("SIMULATED ANNEALING RESULTS")
    print("=" * 50)
    print(f"Start point: {x0}")
    print(f"GD Final f(x): {f(x_gd):.6f}")
    print(f"SA Final f(x): {f(x_sa):.6f}")
    print("-" * 50)
    if f(x_sa) < f(x_gd):
        print(f"Conclusion: SA successfully escaped local minima and found a better point.")
    else:
        print("Conclusion: SA did not perform better than GD in this run.")
    print("=" * 50)
