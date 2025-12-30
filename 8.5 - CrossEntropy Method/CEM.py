import numpy as np
from scipy.stats import multivariate_normal, alpha


class CEM_MvNormal:
    def __init__(self, mu, sigma):
        self.mu = np.array(mu, dtype=float)
        self.sigma = np.array(sigma, dtype=float)

    def sample(self, m):
        samples = multivariate_normal.rvs(mean=self.mu, cov=self.sigma, size=m)
        if m == 1:
            samples = samples.reshape(1, -1)
        return samples

    @classmethod
    def fit(cls, samples, old_dist = None, alpha=0.9):
        # MLE estimate for mean and covariance
        mu = np.mean(samples, axis=0)
        sigma = np.cov(samples, rowvar=False, bias=True)  # bias=True divides by N

        if old_dist is not None:
            mu = alpha * old_dist.mu + (1 - alpha) * mu
            sigma = alpha * old_dist.sigma + (1 - alpha) * sigma

        return cls(mu, sigma)

def cross_entropy_method(f, P, k_max, m=100, m_elite=10, return_history=False):
    history = []

    for k in range(k_max):
        samples = P.sample(m)
        # Evaluate f on each sample
        values = np.array([f(s) for s in samples])

        if return_history:
            history.append({
                'iter': k,
                'distribution': P,
                'samples': samples,
                'values': values
            })

        # Sort indices based on function values
        order = np.argsort(values)

        # Select top m_elite samples
        elite_indices = order[:m_elite]
        elite_samples = samples[elite_indices]

        # Fit new distribution
        P = P.fit(elite_samples, old_dist=P, alpha=0.9)

    if return_history:
        history.append({
            'iter': k_max,
            'distribution': P,
            'samples': None,
            'values': None
        })
        return P, history

    return P