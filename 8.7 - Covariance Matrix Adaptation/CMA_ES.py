import numpy as np

def covariance_matrix_adaptation(f, x, k_max, sigma=1.0, m=None, m_elite=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
    x = np.array(x, dtype=float)
    n = len(x)
    
    # Thiết lập tham số mặc định theo tài liệu
    if m is None:
        m = 4 + int(3 * np.log(n))
    if m_elite is None:
        m_elite = m // 2
    
    # Khởi tạo
    mu = x.copy()

    weights = np.log((m + 1) / 2) - np.log(np.arange(1, m + 1))
    
    weights[:m_elite] /= np.sum(weights[:m_elite])
    
    mu_eff = 1 / np.sum(weights[:m_elite]**2)
    
    c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
    d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
    
    c_Sigma = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
    c_1 = 2 / ((n + 1.3)**2 + mu_eff)
    c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1/mu_eff) / ((n + 2)**2 + mu_eff))
    
    E_norm = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
    
    p_sigma = np.zeros(n)
    p_Sigma = np.zeros(n)
    Sigma = np.eye(n)
    
    # Lịch sử để vẽ biểu đồ
    history = []
    
    for k in range(k_max):
        try:
            D2, B = np.linalg.eigh(Sigma)
            D = np.sqrt(np.maximum(D2, 0))
            inv_sqrt_Sigma = B @ np.diag(1.0/D) @ B.T
            sqrt_Sigma = B @ np.diag(D) @ B.T
        except np.linalg.LinAlgError:
            break

        # Sampling
        samples = []
        z_vectors = [] 
        
        for _ in range(m):
            z = np.random.randn(n)
            step = sigma * (sqrt_Sigma @ z)
            samples.append(mu + step)
            z_vectors.append(z)
            
        samples = np.array(samples)
        
        # Lưu trạng thái vào lịch sử
        history.append({
            'mu': mu.copy(),
            'Sigma': Sigma.copy(),
            'sigma': sigma,
            'samples': samples.copy(),
            'k': k
        })
        
        # Đánh giá và Sắp xếp
        fitness_values = np.array([f(s) for s in samples])
        sorted_indices = np.argsort(fitness_values)
        
        best_samples = samples[sorted_indices]
        
        # Chọn lọc và cập nhật trung bình
        delta_s_elite = (best_samples[:m_elite] - mu) / sigma
        
        delta_w = np.zeros(n)
        for i in range(m_elite):
            delta_w += weights[i] * delta_s_elite[i]
            
        mu_old = mu.copy()
        mu += sigma * delta_w
        
        # Evolution path
        C_delta_w = inv_sqrt_Sigma @ delta_w
        
        p_sigma = (1 - c_sigma) * p_sigma + \
                  np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_delta_w
                  
        sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / E_norm - 1))
        
        # Covariance adaptation
        h_sigma_cond = (np.linalg.norm(p_sigma) / 
                       np.sqrt(1 - (1 - c_sigma)**(2 * (k + 1)))) < (1.4 + 2/(n + 1)) * E_norm
        h_sigma = 1.0 if h_sigma_cond else 0.0
        
        p_Sigma = (1 - c_Sigma) * p_Sigma + \
                  h_sigma * np.sqrt(c_Sigma * (2 - c_Sigma) * mu_eff) * delta_w
        
        # Điều chỉnh trọng số cho cập nhật Rank-mu
        current_delta_s = (best_samples - mu_old) / sigma
        
        w_adjusted = weights.copy()
        for i in range(m):
            if weights[i] < 0:
                norm_C_step = np.linalg.norm(inv_sqrt_Sigma @ current_delta_s[i])
                w_adjusted[i] = weights[i] * n / (norm_C_step**2)

        # Cập nhật Rank-1
        rank_1 = np.outer(p_Sigma, p_Sigma)
        
        # Cập nhật Rank-mu
        rank_mu = np.zeros((n, n))
        for i in range(m):
            step = current_delta_s[i]
            rank_mu += w_adjusted[i] * np.outer(step, step)
            
        # Công thức cập nhật Sigma đầy đủ
        Sigma = (1 - c_1 - c_mu) * Sigma + \
                c_1 * (rank_1 + (1 - h_sigma) * c_Sigma * (2 - c_Sigma) * Sigma) + \
                c_mu * rank_mu
        
        # Đảm bảo tính đối xứng
        Sigma = np.triu(Sigma) + np.triu(Sigma, 1).T

    return mu, history