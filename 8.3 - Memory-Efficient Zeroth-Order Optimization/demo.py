"""
Demo minh họa thuật toán MeZO so với Gradient Descent

Chúng ta sử dụng hàm Rosenbrock: f(x, y) = (a-x)^2 + b(y-x^2)^2
Đây là một hàm thử nghiệm phổ biến, có cực tiểu toàn cục tại (1, 1).
"""

import numpy as np
import matplotlib.pyplot as plt
from MeZO import mezo_optimizer, gradient_descent
from matplotlib.lines import Line2D

# =============================================================================
# Định nghĩa hàm Rosenbrock
# =============================================================================

def rosenbrock(x: np.ndarray) -> float:
    a, b = 1, 100
    return (a - x[0])**2 + b * (x[1] - x[0]**2)**2

def grad_rosenbrock(x: np.ndarray) -> np.ndarray:
    a, b = 1, 100
    grad = np.zeros_like(x)
    grad[0] = -2 * (a - x[0]) - 4 * b * x[0] * (x[1] - x[0]**2)
    grad[1] = 2 * b * (x[1] - x[0]**2)
    return grad

# =============================================================================
# Visualization
# =============================================================================

def plot_comparison(history_gd, history_mezo, x0):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Tạo grid cho contour
    limit = 2.0
    x_range = np.linspace(-limit, limit, 200)
    y_range = np.linspace(-0.5, 3.0, 200)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.array([[rosenbrock(np.array([x, y])) for x in x_range] for y in y_range])
    
    # Vẽ contour
    cp = ax.contourf(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='viridis', alpha=0.6)
    plt.colorbar(cp, label='f(x, y)')
    
    # Đường đi của GD
    path_gd = np.array(history_gd)
    ax.plot(path_gd[:, 0], path_gd[:, 1], 'c-', linewidth=2, label='GD Path')
    ax.scatter(path_gd[-1, 0], path_gd[-1, 1], c='cyan', s=100, marker='X', edgecolors='black', label='GD Final')
    
    # Đường đi của MeZO
    path_mezo = np.array(history_mezo)
    ax.plot(path_mezo[:, 0], path_mezo[:, 1], 'r-', linewidth=1, alpha=0.8, label='MeZO Path')
    ax.scatter(path_mezo[-1, 0], path_mezo[-1, 1], c='red', s=100, marker='X', edgecolors='white', label='MeZO Final')
    
    # Điểm bắt đầu và tối ưu
    ax.plot(x0[0], x0[1], 'w*', markersize=15, label='Start', markeredgecolor='black')
    ax.plot(1, 1, 'g*', markersize=15, label='Optimal (1,1)', markeredgecolor='white')
    
    ax.set_title("Gradient Descent vs MeZO on Rosenbrock")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig("mezo_contour.png")
    plt.show()

def plot_convergence(f_gd, f_mezo):
    plt.figure(figsize=(10, 6))
    plt.plot(f_gd, 'c-', label=f'Gradient Descent (Final: {f_gd[-1]:.4f})')
    plt.plot(f_mezo, 'r-', label=f'MeZO (Final: {f_mezo[-1]:.4f})')
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Function Value (log scale)')
    plt.title('MeZO: Convergence Rate')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig("mezo_convergence.png")
    plt.show()

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Tham số khởi tạo
    x0 = np.array([-1.5, 2.0])
    iters = 1200
    
    # Chạy Gradient Descent
    # Rosenbrock cần alpha rất nhỏ để không hội tụ phân kỳ
    x_gd, hist_gd, f_hist_gd = gradient_descent(
        rosenbrock, grad_rosenbrock, x0, alpha=0.001, max_iter=iters
    )
    
    # Chạy MeZO
    # MeZO có thể cần alpha lớn hơn một chút vì nó là ước lượng ngẫu nhiên
    x_mezo, hist_mezo, f_hist_mezo = mezo_optimizer(
        rosenbrock, x0, alpha=0.001, delta=0.01, max_iter=iters, seed=36
    )
    
    print(f"Result after {iters} iterations:")
    print(f"Start point: {x0}")
    print(f"GD Final f(x): {f_hist_gd[-1]:.6f} tại {x_gd}")
    print(f"MeZO Final f(x): {f_hist_mezo[-1]:.6f} tại {x_mezo}")
    
    # Vẽ biểu đồ
    plot_comparison(hist_gd, hist_mezo, x0)
    plot_convergence(f_hist_gd, f_hist_mezo)