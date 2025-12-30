"""
Memory-Efficient Zeroth-Order (MeZO) Optimizer

MeZO là một thuật toán tối ưu hóa bậc không (zeroth-order) tiết kiệm bộ nhớ, 
đặc biệt hiệu quả cho các mô hình lớn. Thay vì tính toán gradient đầy đủ 
(backpropagation), MeZO xấp xỉ bước nhảy gradient bằng cách sử dụng 
sai phân hữu hạn dọc theo một hướng ngẫu nhiên.

Công thức cập nhật:
    x' = x - alpha * [(f(x + delta*z) - f(x - delta*z)) / (2*delta)] * z

Trong đó:
    - z: Vectơ nhiễu ngẫu nhiên lấy từ phân phối chuẩn tắc N(0, I).
    - delta: Hệ số sai phân hữu hạn (vô hướng).
    - alpha: Hệ số bước nhảy (learning rate).
"""

import numpy as np
from typing import Callable, List, Tuple, Optional

def mezo_optimizer(
    f: Callable[[np.ndarray], float],
    x0: np.ndarray,
    alpha: float = 1e-2,
    delta: float = 1e-3,
    max_iter: int = 1000,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, List[np.ndarray], List[float]]:
    """
    Thuật toán MeZO để tối ưu hóa hàm mục tiêu.
    
    Parameters
    ----------
    f : Callable
        Hàm mục tiêu cần tối ưu hóa (chỉ trả về giá trị số).
    x0 : np.ndarray
        Điểm khởi đầu.
    alpha : float
        Hệ số bước nhảy (learning rate).
    delta : float
        Đại lượng vô hướng cho sai phân hữu hạn.
    max_iter : int
        Số vòng lặp tối đa.
    seed : int, optional
        Seed để tái lập kết quả nhiễu ngẫu nhiên.
        
    Returns
    -------
    x_best : np.ndarray
        Điểm tốt nhất tìm được.
    history : List[np.ndarray]
        Lịch sử các điểm x qua từng bước.
    f_history : List[float]
        Lịch sử giá trị f(x) qua các vòng lặp.
    """
    if seed is not None:
        np.random.seed(seed)
        
    x = x0.copy().astype(float)
    n = len(x)
    
    history = [x.copy()]
    f_history = [f(x)]
    
    x_best = x.copy()
    y_best = f_history[0]

    for k in range(1, max_iter + 1):
        # 1. Lấy hướng ngẫu nhiên z từ phân phối chuẩn tắc
        z = np.random.randn(n)
        
        # 2. Tính toán hai giá trị hàm tại điểm nhiễu (f-cập nhật tại chỗ xấp xỉ)
        y_plus = f(x + delta * z)
        y_minus = f(x - delta * z)
        
        # 3. Ước lượng gradient theo hướng z (Projected Gradient)
        # grad_approx = (f(x + delta*z) - f(x - delta*z)) / (2 * delta)
        grad_estimate = (y_plus - y_minus) / (2 * delta)
        
        # 4. Cập nhật tại chỗ
        x = x - alpha * grad_estimate * z
        
        # Lưu trữ lịch sử và cập nhật điểm tốt nhất
        current_f = f(x)
        if current_f < y_best:
            y_best = current_f
            x_best = x.copy()
            
        history.append(x.copy())
        f_history.append(current_f)
        
    return x_best, history, f_history

def gradient_descent(
    f, grad_f, x0, alpha=0.01, max_iter=100
):
    """Sử dụng lại GD để so sánh"""
    x = x0.copy().astype(float)
    history = [x.copy()]
    f_history = [f(x)]
    
    for _ in range(max_iter):
        g = grad_f(x)
        x = x - alpha * g
        history.append(x.copy())
        f_history.append(f(x))
    return x, history, f_history