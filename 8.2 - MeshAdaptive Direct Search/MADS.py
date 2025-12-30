import numpy as np

def rand_positive_spanning_set(alpha, n):
    delta = int(round(1.0 / np.sqrt(alpha)))
    if delta == 0: delta = 1

    # Tạo ma trận L
    diag_vals = delta * np.random.choice([1, -1], size=n)
    L = np.diag(diag_vals)
    
    # Điền các phần tử dưới đường chéo chính
    for i in range(1, n):
        for j in range(i):
            low, high = -delta + 1, delta - 1
            if low <= high:
                L[i, j] = np.random.randint(low, high + 1)
            else:
                L[i, j] = 0
    
    # Hoán vị hàng và cột để tạo D
    p_row = np.random.permutation(n)
    p_col = np.random.permutation(n)
    D = L[p_row, :][:, p_col]

    # Thêm hướng tổng âm (hướng thứ n+1)
    d_last = -np.sum(D, axis=1).reshape(-1, 1)
    
    return np.hstack((D, d_last))

def mesh_adaptive_direct_search(f, x0, epsilon=1e-4, max_iter=1000, seed=None):
    if seed is not None:
        np.random.seed(seed)
    x = np.array(x0, dtype=float)
    n = len(x)
    alpha = 1.0
    y = f(x)
    
    path = [x.copy()]
    
    for _ in range(max_iter):
        if alpha <= epsilon:
            break
        
        improved = False
        directions = rand_positive_spanning_set(alpha, n)
        
        for d in directions.T:
            x_prime = x + alpha * d
            y_prime = f(x_prime)
            
            if y_prime < y:
                # Cập nhật điểm hiện tại
                x, y, improved = x_prime, y_prime, True
                
                # Extension step
                x_prime_ext = x + 3 * alpha * d
                y_prime_ext = f(x_prime_ext)
                
                if y_prime_ext < y:
                    x, y = x_prime_ext, y_prime_ext
                
                # Opportunistic
                path.append(x.copy())
                break
        
        # Cập nhật alpha
        if improved:
            alpha = min(1.0, 4 * alpha)
        else:
            alpha = alpha / 4.0
            
    return x, np.array(path)