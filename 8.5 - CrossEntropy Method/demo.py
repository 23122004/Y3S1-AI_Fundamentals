import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import os

from CEM import cross_entropy_method, CEM_MvNormal

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

def sphere_function(x):
    return np.sum(x ** 2, axis=-1)

def run_experiment():
    mu_init = [5.0, 5.0]
    sigma_init = 2
    cov_init = [[sigma_init**2, 0.0], [0.0, sigma_init**2]]

    k_max = 100
    m = 50

    np.random.seed(42)
    P_cem = CEM_MvNormal(mu_init, cov_init)
    _, history_cem = cross_entropy_method(
        sphere_function, P_cem, k_max, m=m, m_elite=5, return_history=True
    )

    # Visualization
    iters_to_plot = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90]

    # Create Grid for Contour
    grid_x = np.linspace(-10, 10, 150)
    grid_y = np.linspace(-10, 10, 150)
    X, Y = np.meshgrid(grid_x, grid_y)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i,j] = sphere_function(np.array([X[i,j], Y[i,j]]))

    def create_grid_plot(history, title_prefix, filename):
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        axes = axes.flatten()

        for idx, iter_idx in enumerate(iters_to_plot):
            ax = axes[idx]
            ax.set_aspect('equal', adjustable='box')

            # Contour Background
            ax.contourf(X, Y, Z, levels=30, cmap='viridis')

            if iter_idx < len(history):
                record = history[iter_idx]
                dist = record['distribution']
                samples = record['samples']

                # Plot Samples
                if samples is not None:
                    ax.scatter(samples[:, 0], samples[:, 1], c='white', s=5, alpha=0.7)

                # Plot Covariance Ellipse
                std_levels = [1.0, 2.0, 3.0]
                line_widths = [1.5, 1.0, 0.8]
                for n_std, lw in zip(std_levels, line_widths):
                    plot_confidence_ellipse(
                        dist.mu, dist.sigma, ax,
                        n_std=n_std,
                        edgecolor='white',
                        lw=lw,
                        linestyle='-'
                    )

                # Mark Mean
                ax.plot(dist.mu[0], dist.mu[1], 'r+', markersize=8, markeredgewidth=2)

            ax.set_title(f"Iter: {iter_idx}", fontsize=10)
            ax.set_xlim(-10, 10)
            ax.set_ylim(-10, 10)
            ax.set_xticks([])
            ax.set_yticks([])

        # plt.suptitle(f"{title_prefix} on sphere Function", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(filename)
        plt.close()
        print(f"Saved {filename}")

    create_grid_plot(history_cem, "Cross-Entropy Method", "CEM.png")

if __name__ == "__main__":
    run_experiment()
