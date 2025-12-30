"""
Noisy Descent Optimizer

Noisy Descent là kỹ thuật cải tiến Gradient Descent bằng cách thêm nhiễu Gaussian 
vào quá trình cập nhật tham số. Điều này giúp thoát khỏi các điểm yên ngựa (saddle points).

Công thức cập nhật:
    x^(k+1) = x^(k) + d^(k) + ε^(k)

Trong đó:
    - d^(k) = -α * ∇f(x^(k)) là bước gradient descent
    - ε^(k) ~ N(0, σ²) là nhiễu Gaussian
    - σ thường giảm dần theo thời gian (ví dụ: σ/k)
"""

import numpy as np
from typing import Callable, List, Tuple, Optional


def noisy_descent(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    alpha: float = 0.1,
    sigma: float = 0.1,
    max_iter: int = 100,
    decay_sigma: bool = True,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, List[np.ndarray], List[float]]:
    """
    Thuật toán Noisy Descent để tối ưu hóa hàm mục tiêu.
    
    Parameters
    ----------
    f : Callable
        Hàm mục tiêu cần tối ưu hóa.
    grad_f : Callable
        Hàm gradient của hàm mục tiêu.
    x0 : np.ndarray
        Điểm khởi đầu.
    alpha : float
        Hệ số learning rate (step size).
    sigma : float
        Độ lệch chuẩn ban đầu của nhiễu Gaussian.
    max_iter : int
        Số vòng lặp tối đa.
    decay_sigma : bool
        Nếu True, độ lệch chuẩn nhiễu sẽ giảm dần theo thời gian (σ/k).
    seed : int, optional
        Seed cho bộ sinh số ngẫu nhiên.
        
    Returns
    -------
    x_best : np.ndarray
        Điểm tối ưu tìm được.
    history : List[np.ndarray]
        Lịch sử các điểm đã duyệt qua.
    f_history : List[float]
        Lịch sử giá trị hàm mục tiêu.
    """
    if seed is not None:
        np.random.seed(seed)
    
    x = x0.copy().astype(float)
    n = len(x)
    
    history = [x.copy()]
    f_history = [f(x)]
    
    x_best = x.copy()
    f_best = f(x)
    
    for k in range(1, max_iter + 1):
        # Tính gradient
        g = grad_f(x)
        
        # Bước gradient descent
        d = -alpha * g
        
        # Tính độ lệch chuẩn nhiễu (có thể giảm dần)
        current_sigma = sigma / k if decay_sigma else sigma
        
        # Thêm nhiễu Gaussian
        epsilon = current_sigma * np.random.randn(n)
        
        # Cập nhật x theo công thức Noisy Descent
        x = x + d + epsilon
        
        # Lưu lịch sử
        history.append(x.copy())
        f_val = f(x)
        f_history.append(f_val)
        
        # Cập nhật điểm tốt nhất
        if f_val < f_best:
            x_best = x.copy()
            f_best = f_val
    
    return x_best, history, f_history


def gradient_descent(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    alpha: float = 0.1,
    max_iter: int = 100
) -> Tuple[np.ndarray, List[np.ndarray], List[float]]:
    """
    Thuật toán Gradient Descent tiêu chuẩn (không có nhiễu) để so sánh.
    
    Parameters
    ----------
    f : Callable
        Hàm mục tiêu cần tối ưu hóa.
    grad_f : Callable
        Hàm gradient của hàm mục tiêu.
    x0 : np.ndarray
        Điểm khởi đầu.
    alpha : float
        Hệ số learning rate (step size).
    max_iter : int
        Số vòng lặp tối đa.
        
    Returns
    -------
    x_final : np.ndarray
        Điểm tối ưu tìm được.
    history : List[np.ndarray]
        Lịch sử các điểm đã duyệt qua.
    f_history : List[float]
        Lịch sử giá trị hàm mục tiêu.
    """
    x = x0.copy().astype(float)
    
    history = [x.copy()]
    f_history = [f(x)]
    
    for k in range(max_iter):
        # Tính gradient
        g = grad_f(x)
        
        # Cập nhật x (gradient descent thuần túy)
        x = x - alpha * g
        
        # Lưu lịch sử
        history.append(x.copy())
        f_history.append(f(x))
    
    return x, history, f_history
