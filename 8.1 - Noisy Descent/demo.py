"""
Demo minh họa thuật toán Noisy Descent với hàm Rastrigin

Hàm Rastrigin là một hàm kiểm thử tối ưu hóa nổi tiếng với nhiều điểm cực tiểu
cục bộ, được sử dụng để đánh giá khả năng thoát khỏi local minima của các
thuật toán tối ưu hóa.

Hàm Rastrigin:  f(x) = An + Σ[x_i² - A·cos(2πx_i)]
với A = 10, điểm cực tiểu toàn cục tại (0, 0) với f(0,0) = 0
"""

import numpy as np
import matplotlib.pyplot as plt
from NoisyDescent import noisy_descent, gradient_descent

# =============================================================================
# Định nghĩa hàm Rastrigin 
# =============================================================================

A_RASTRIGIN = 10  # Hằng số Rastrigin

def rastrigin(x: np.ndarray) -> float:
    """
    Hàm Rastrigin: f(x) = An + Σ[x_i² - A·cos(2πx_i)]
    
    - Điểm cực tiểu toàn cục: x* = (0, 0, ..., 0) với f(x*) = 0
    - Có rất nhiều điểm cực tiểu cục bộ (khoảng 10^n cho n chiều)
    - Thường dùng trong phạm vi [-5.12, 5.12]^n
    """
    n = len(x)
    return float(A_RASTRIGIN * n + np.sum(x**2 - A_RASTRIGIN * np.cos(2 * np.pi * x)))

def grad_rastrigin(x: np.ndarray) -> np.ndarray:
    """
    Gradient của hàm Rastrigin:
    ∂f/∂x_i = 2x_i + 2πA·sin(2πx_i)
    """
    return 2 * x + 2 * np.pi * A_RASTRIGIN * np.sin(2 * np.pi * x)

def f(x: np.ndarray) -> float:
    return rastrigin(x)

def grad_f(x: np.ndarray) -> np.ndarray:
    return grad_rastrigin(x)

# =============================================================================
# Hàm tìm seed tốt nhất
# =============================================================================

def find_best_seed(
    f_func,
    grad_f_func,
    x0: np.ndarray,
    alpha: float = 0.01,
    sigma: float = 4.0,
    max_iter: int = 300,
    decay_sigma: bool = True,
    seed_range: range = range(100)
) -> tuple:
    """
    Tìm seed tốt nhất.
    
    Returns
    -------
    best_seed : int
    best_f_val : float
    best_x : np.ndarray
    """
    best_seed = None
    best_f_val = float('inf')
    best_x = None
    
    for seed in seed_range:
        x_result, _, _ = noisy_descent(
            f_func, grad_f_func, x0,
            alpha=alpha, sigma=sigma,
            max_iter=max_iter, decay_sigma=decay_sigma,
            seed=seed
        )
        f_val = f_func(x_result)
        
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
    history_nd: list,
    x0: np.ndarray,
    title: str = "GD vs Noisy Descent Comparison",
    x_range: tuple = (-4, 4),
    save_path: str = "contour_paths.png",
    show_plot: bool = True
) -> None:
    """
    Vẽ contour plot với đường đi của cả 2 thuật toán.
    
    Args:
        objective_func: Hàm mục tiêu f(x)
        history_gd: Lịch sử các điểm của Gradient Descent
        history_nd: Lịch sử các điểm của Noisy Descent
        x0: Điểm khởi đầu
        title: Tiêu đề
        x_range: Phạm vi vẽ contour (min, max)
        save_path: Đường dẫn lưu file ảnh
        show_plot: Có hiển thị plot không
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Tạo grid cho contour plot
    x1_range = np.linspace(x_range[0], x_range[1], 201)
    x2_range = np.linspace(x_range[0], x_range[1], 201)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    Z = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Z[i, j] = objective_func(np.array([X1[i, j], X2[i, j]]))
    
    # Vẽ contour filled với colormap đẹp
    contourf = ax.contourf(X1, X2, Z, levels=50, cmap='viridis', alpha=0.8)
    ax.contour(X1, X2, Z, levels=20, colors='white', linewidths=0.5, alpha=0.5)
    plt.colorbar(contourf, ax=ax, label='f(x)')
    
    # Vẽ đường đi của Gradient Descent
    pts_gd = np.array(history_gd)
    ax.plot(pts_gd[:, 0], pts_gd[:, 1], 'c-', linewidth=1.5, alpha=0.7)
    ax.scatter(pts_gd[::10, 0], pts_gd[::10, 1], c='cyan', s=30, 
               edgecolors='white', linewidths=1, label='GD path', zorder=4)
    ax.scatter(pts_gd[-1, 0], pts_gd[-1, 1], c='blue', s=120, marker='X',
               edgecolors='white', linewidths=2, label='GD final', zorder=5)
    
    # Vẽ đường đi của Noisy Descent
    pts_nd = np.array(history_nd)
    ax.plot(pts_nd[:, 0], pts_nd[:, 1], 'r-', linewidth=1.5, alpha=0.7)
    ax.scatter(pts_nd[::10, 0], pts_nd[::10, 1], c='red', s=30, 
               edgecolors='white', linewidths=1, label='ND path', zorder=4)
    ax.scatter(pts_nd[-1, 0], pts_nd[-1, 1], c='darkred', s=120, marker='X',
               edgecolors='white', linewidths=2, label='ND final', zorder=5)
    
    # Đánh dấu điểm bắt đầu và điểm tối ưu toàn cục (kích thước nhỏ hơn trên plot)
    ax.plot(x0[0], x0[1], 'w*', markersize=15, markeredgecolor='black', 
            markeredgewidth=1.5, zorder=6)
    ax.plot(0, 0, 'g*', markersize=15, markeredgecolor='white', 
            markeredgewidth=1.5, zorder=6)
    
    # Tạo legend entries riêng cho Start và Global Optimum với kích thước nhỏ
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan', 
               markersize=8, markeredgecolor='white', label='GD path'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='blue', 
               markersize=10, markeredgecolor='white', label='GD final'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
               markersize=8, markeredgecolor='white', label='ND path'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='darkred', 
               markersize=10, markeredgecolor='white', label='ND final'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='white', 
               markersize=10, markeredgecolor='black', label='Start'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='green', 
               markersize=10, markeredgecolor='white', label='Global Opt (0,0)'),
    ]
    
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_title(f'{title}\n(multiple local minima)', fontsize=14)
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, ncol=2)
    ax.set_aspect('equal')
    
    # Giới hạn vùng hiển thị
    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(x_range[0], x_range[1])
    ax.grid(True, alpha=0.3, color='white')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {save_path}")
    
    if show_plot:
        plt.show()
    
    plt.close(fig)


def plot_convergence(
    f_history_gd: list,
    f_history_nd: list,
    title: str = "Objective function value vs Iterations",
    save_path: str = "convergence.png",
    show_plot: bool = True
) -> None:
    """
    Vẽ biểu đồ hội tụ của cả 2 thuật toán.
    
    Args:
        f_history_gd: Lịch sử giá trị hàm của Gradient Descent
        f_history_nd: Lịch sử giá trị hàm của Noisy Descent
        title: Tiêu đề
        save_path: Đường dẫn lưu file ảnh
        show_plot: Có hiển thị plot không
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    iterations = range(len(f_history_gd))
    ax.semilogy(iterations, f_history_gd, 'b-', linewidth=2.5, 
                label='Gradient Descent', alpha=0.9)
    ax.semilogy(iterations, f_history_nd, 'r-', linewidth=2.5, 
                label='Noisy Descent', alpha=0.9)
    
    # Đánh dấu giá trị cuối cùng
    ax.axhline(y=f_history_gd[-1], color='blue', linestyle='--', alpha=0.5,
               label=f'GD final: {f_history_gd[-1]:.2f}')
    ax.axhline(y=f_history_nd[-1], color='red', linestyle='--', alpha=0.5,
               label=f'ND final: {f_history_nd[-1]:.2f}')
    
    # Đánh dấu điểm cuối
    ax.scatter(len(f_history_gd)-1, f_history_gd[-1], c='blue', s=100, 
               marker='o', edgecolors='white', linewidths=2, zorder=5)
    ax.scatter(len(f_history_nd)-1, f_history_nd[-1], c='red', s=100, 
               marker='o', edgecolors='white', linewidths=2, zorder=5)
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('$f(x)$ (log scale)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10, loc='lower left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {save_path}")
    
    if show_plot:
        plt.show()
    
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    # Điểm khởi đầu 
    x0 = np.array([2.5, 2.5])

    # Tham số
    alpha = 0.01   # Learning rate
    sigma = 4      # Độ lệch chuẩn của nhiễu
    max_iter = 300 # Số vòng lặp

    # best_seed, best_f_val, best_x = find_best_seed(
    #     f, grad_f, x0,
    #     alpha=0.01, sigma=4, max_iter=200,
    #     seed_range=range(100)
    # )

    # Gradient Descent
    x_gd, history_gd, f_history_gd = gradient_descent(
        f, grad_f, x0, alpha=alpha, max_iter=max_iter
    )

    # Noisy Descent
    # print("Best seed:", best_seed)
    x_nd, history_nd, f_history_nd = noisy_descent(
        f, grad_f, x0, alpha=alpha, sigma=sigma, max_iter=max_iter, 
        decay_sigma=True, seed=50 # best_seed
    )

    # Visualization 

    # Fig 1: Contour plot
    plot_contour_paths(
        objective_func=f,
        history_gd=history_gd,
        history_nd=history_nd,
        x0=x0,
        title="Gradient Descent vs Noisy Descent on Rastrigin",
        x_range=(-4, 4),
        save_path="noisy_descent_contour.png",
        show_plot=True
    )

    # Fig 2: Convergence plot
    plot_convergence(
        f_history_gd=f_history_gd,
        f_history_nd=f_history_nd,
        title="Noisy Descent: Convergence Rate",
        save_path="noisy_descent_convergence.png",
        show_plot=True
    )

    # In kết quả

    print("\n" + "=" * 50)
    print("NOISY DESCENT RESULTS ON RASTRIGIN FUNCTION")
    print("=" * 50)

    print(f"Settings:")
    print(f"  Start: {x0}, alpha: {alpha}, sigma: {sigma}, iter: {max_iter}")

    print("-" * 50)
    print(f"Gradient Descent final f(x): {f(x_gd):.6f}")
    print(f"Noisy Descent final f(x): {f(x_nd):.6f}")

    # Concise conclusion
    print("-" * 50)
    if f(x_nd) < f(x_gd):
        improvement = ((f(x_gd) - f(x_nd)) / f(x_gd)) * 100
        print(f"Conclusion: Noisy Descent performed better ({improvement:.2f}% improvement).")
    else:
        print("Conclusion: Noisy Descent results were comparable or worse than GD.")
    print("=" * 50)