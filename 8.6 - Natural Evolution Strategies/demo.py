import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import os

from scipy.stats import multivariate_normal
from NES import natural_evolution_strategies, NES_MvNormal

def plot_confidence_ellipse(mean, cov, ax, n_std=2.0, facecolor='none', **kwargs):
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean[0], mean[1])

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def wheeler_ridge_function(x):
    x1, x2 = x[0], x[1]
    return -np.exp(-(x1*x2 - 1.5)**2 - (x2 - 1.5)**2)

def create_grid_plot(history, iters_to_plot, filename):
    fig, axes = plt.subplots(3, 4, figsize=(16, 12), facecolor='white')
    axes = axes.flatten()

    x_lim = (0, 3)
    y_lim = (0, 3)
    res = 100

    x = np.linspace(x_lim[0], x_lim[1], res)
    y = np.linspace(y_lim[0], y_lim[1], res)
    X, Y = np.meshgrid(x, y)

    Z_f = np.zeros_like(X)
    for i in range(res):
        for j in range(res):
            Z_f[i, j] = -wheeler_ridge_function([X[i, j], Y[i, j]])

    for idx, iter_idx in enumerate(iters_to_plot):
        ax = axes[idx]
        im = ax.contourf(X, Y, Z_f, levels=50, cmap='viridis', alpha=0.9)

        if iter_idx < len(history):
            record = history[iter_idx]
            dist = record['distribution']
            samples = record['samples']

            try:
                rv = multivariate_normal(dist.mu, dist.sigma)
                Z_pdf = rv.pdf(np.dstack((X, Y)))
                ax.contour(X, Y, Z_pdf, levels=5, colors='white', linewidths=1.0, alpha=0.8)
            except:
                pass

            if samples is not None:
                ax.scatter(samples[:, 0], samples[:, 1],
                           color='white', s=20, alpha=0.8, edgecolors='none')

            ax.scatter(dist.mu[0], dist.mu[1], color='red', marker='+', s=50, linewidths=1.5)

        ax.set_title(f"Iter: {iter_idx}", fontsize=11, fontweight='bold')
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_aspect('equal')

        if idx % 4 != 0: ax.set_yticklabels([])
        if idx < 8: ax.set_xticklabels([])

        ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.show()
    print(f"Saved: {filename}")

def run_experiment():
    np.random.seed(0)
    A_init = np.eye(2) * 0.5
    mu_init = np.array([2.0, 2.0])
    alpha = 0.01
    k_max = 450
    m = 40

    P_nes = NES_MvNormal(mu_init, A_init)

    _, history_nes = natural_evolution_strategies(
        wheeler_ridge_function, P_nes, k_max, m=m, alpha=alpha, return_history=True
    )

    iters_to_plot = [0, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440]

    create_grid_plot(history_nes, iters_to_plot, "NES.png")

if __name__ == "__main__":
    run_experiment()
