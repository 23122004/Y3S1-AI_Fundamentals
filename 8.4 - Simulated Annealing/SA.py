"""
Simulated Annealing (SA) Optimizer

Simulated Annealing là thuật toán tối ưu hóa lấy cảm hứng từ quá trình luyện kim.
Nó sử dụng "nhiệt độ" để kiểm soát khả năng chấp nhận các bước đi làm tăng giá trị hàm mục tiêu,
giúp thuật toán thoát khỏi các cực tiểu cục bộ.

Tiêu chuẩn Metropolis:
    P(chấp nhận) = 1 nếu Δy <= 0
    P(chấp nhận) = exp(-Δy / t) nếu Δy > 0

Trong đó:
    - Δy = f(x') - f(x)
    - t là nhiệt độ hiện tại
"""

import numpy as np
from typing import Callable, List, Tuple, Optional, Union


def simulated_annealing(
    f: Callable[[np.ndarray], float],
    x0: np.ndarray,
    t0: float = 10.0,
    sigma: float = 1.0,
    max_iter: int = 1000,
    schedule: str = 'exponential',
    gamma: float = 0.95,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, List[np.ndarray], List[float]]:
    """
    Thuật toán Simulated Annealing để tối ưu hóa hàm mục tiêu.
    
    Parameters
    ----------
    f : Callable
        Hàm mục tiêu cần tối ưu hóa.
    x0 : np.ndarray
        Điểm khởi đầu.
    t0 : float
        Nhiệt độ ban đầu.
    sigma : float
        Độ lệch chuẩn của phân phối đề xuất (Gaussian).
    max_iter : int
        Số vòng lặp tối đa.
    schedule : str
        Lịch trình hạ nhiệt: 'exponential', 'fast', 'logarithmic'.
    gamma : float
        Hệ số suy giảm cho lịch trình 'exponential'.
    seed : int, optional
        Seed cho bộ sinh số ngẫu nhiên.
        
    Returns
    -------
    x_best : np.ndarray
        Điểm tốt nhất tìm được trong toàn bộ quá trình.
    history : List[np.ndarray]
        Lịch sử các điểm x tại mỗi bước.
    f_history : List[float]
        Lịch sử các giá trị f(x) tốt nhất tìm được đến thời điểm k.
    """
    if seed is not None:
        np.random.seed(seed)
    
    x = x0.copy().astype(float)
    n = len(x)
    
    y = f(x)
    x_best = x.copy()
    y_best = y
    
    history = [x.copy()]
    f_history = [y_best]
    
    t = t0
    
    for k in range(1, max_iter + 1):
        # Đề xuất bước đi mới (Neighbor selection)
        x_new = x + sigma * np.random.randn(n)
        y_new = f(x_new)
        
        # Tính độ chênh lệch
        delta_y = y_new - y
        
        # Tiêu chuẩn Metropolis
        if delta_y <= 0 or np.random.rand() < np.exp(-delta_y / t):
            x = x_new
            y = y_new
            
        # Cập nhật điểm tốt nhất
        if y < y_best:
            x_best = x.copy()
            y_best = y
            
        # Cập nhật nhiệt độ theo lịch trình
        if schedule == 'exponential':
            t = t0 * (gamma ** k)
        elif schedule == 'fast':
            t = t0 / k
        elif schedule == 'logarithmic':
            t = t0 * np.log(2) / np.log(k + 1)
            
        # Lưu lịch sử
        history.append(x.copy())
        f_history.append(y_best)
        
    return x_best, history, f_history

# Định nghĩa gradient_descent
def gradient_descent(
    f, grad_f, x0, alpha=0.1, max_iter=100
):
    x = x0.copy().astype(float)
    history = [x.copy()]
    f_history = [f(x)]
    
    for _ in range(max_iter):
        g = grad_f(x)
        x = x - alpha * g
        history.append(x.copy())
        f_history.append(f(x))
    return x, history, f_history