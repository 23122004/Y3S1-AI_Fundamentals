import copy
import numpy as np

class NES_MvNormal:
    def __init__(self, mu, A):
        self.mu = np.array(mu, dtype=float)
        self.A = np.array(A, dtype=float)
        self.update_sigma()

    def update_sigma(self):
        self.sigma = self.A.T @ self.A
        self.sigma_inv_reg = np.linalg.inv(self.sigma + 0.2 * np.eye(len(self.mu)))

    def sample(self, m):
        return np.random.multivariate_normal(self.mu, self.sigma, size=m)

    def compute_gradients(self, x):
        diff = x - self.mu

        grad_mu = self.sigma_inv_reg @ diff
        grad_sigma = 0.5 * (np.outer(grad_mu, grad_mu) - self.sigma_inv_reg)
        grad_A = self.A @ (grad_sigma + grad_sigma.T)

        return grad_mu, grad_A

    def update(self, avg_grad_mu, avg_grad_A, alpha):
        self.mu -= alpha * avg_grad_mu
        self.A -= alpha * avg_grad_A
        self.update_sigma()

def natural_evolution_strategies(f, theta, k_max, m=100, alpha=0.01, return_history=False):
    history = []

    for k in range(k_max):
        samples = theta.sample(m)
        if m == 1:
            samples = samples.reshape(1, -1)

        if return_history:
            history.append({
                'iter': k,
                'distribution': copy.deepcopy(theta),
                'samples': samples
            })

        # Accumulate gradients
        total_grad_mu = np.zeros_like(theta.mu)
        total_grad_A = np.zeros_like(theta.A)

        for x in samples:
            val = f(x)
            g_mu, g_A = theta.compute_gradients(x)
            total_grad_mu += val * g_mu
            total_grad_A += val * g_A

        # Average
        avg_grad_mu = total_grad_mu / m
        avg_grad_A = total_grad_A / m

        # Update
        theta.update(avg_grad_mu, avg_grad_A, alpha)

    if return_history:
        history.append({
            'iter': k_max,
            'distribution': copy.deepcopy(theta),
            'samples': None
        })
        return theta, history

    return theta