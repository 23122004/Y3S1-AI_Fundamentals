import numpy as np
import matplotlib.pyplot as plt
from MADS import rand_positive_spanning_set

def ackley_function(x):
    x = np.array(x)
    d = len(x)
    a, b, c = 20, 0.2, 2 * np.pi
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(c * x))
    term1 = -a * np.exp(-b * np.sqrt(sum_sq / d))
    term2 = -np.exp(sum_cos / d)
    return term1 + term2 + a + np.exp(1)

def run_demo():
    x = np.array([1.5, 1.5]) 
    n = len(x)
    alpha = 1.0
    y = ackley_function(x)
    
    x_range = np.linspace(-1, 4, 100)
    y_range = np.linspace(-1, 4, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.array([[ackley_function([xi, yi]) for xi, yi in zip(xr, yr)] for xr, yr in zip(X, Y)])

    fig, axes = plt.subplots(3, 4, figsize=(15, 11))
    axes = axes.flatten()
    
    for i in range(12):
        ax = axes[i]
        ax.contour(X, Y, Z, levels=15, cmap='viridis', alpha=0.1)
        
        directions = rand_positive_spanning_set(alpha, n)
        ax.plot(x[0], x[1], 'ko', markersize=4, label='Current $x^{(k)}$' if i==0 else "")
        
        improved = False
        best_d = None
        
        for d in directions.T:
            x_prime = x + alpha * d
            y_prime = ackley_function(x_prime)
            ax.plot([x[0], x_prime[0]], [x[1], x_prime[1]], 'k--', lw=0.5, alpha=0.3)
            
            if not improved and y_prime < y:
                x_next, y_next = x_prime, y_prime
                improved = True
                best_d = d
        
        if improved:
            ax.arrow(x[0], x[1], alpha*best_d[0], alpha*best_d[1], 
                     head_width=0.08, head_length=0.08, fc='blue', ec='blue', alpha=0.6)
            
            x_ext = x + 4 * alpha * best_d
            y_ext = ackley_function(x_ext)
            ax.plot(x_ext[0], x_ext[1], 'rx', markersize=5, label='Extension' if i==0 else "")
            
            if y_ext < y_next:
                x, y = x_ext, y_ext
            else:
                x, y = x_next, y_next
            
            alpha = min(1.0, 4 * alpha)
        else:
            alpha = alpha / 4.0
            
        ax.set_title(f"Iter {i+1}: $\\alpha={alpha:.3f}$", fontsize=9)
        ax.set_xlim(-1, 4)
        ax.set_ylim(-1, 4)
        ax.tick_params(labelsize=8)
        if i == 0:
            ax.legend(fontsize=7, loc='upper right')

    plt.subplots_adjust(wspace=0.3, hspace=0.4, left=0.07, right=0.95, top=0.92, bottom=0.05)
    plt.suptitle("MADS Optimization demo", fontsize=14)
    plt.show()

if __name__ == "__main__":
    np.random.seed(42)
    run_demo()