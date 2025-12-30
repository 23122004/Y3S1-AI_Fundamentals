import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from CMA_ES import covariance_matrix_adaptation

def flower_function(x):
    """
    Hàm flower mô phỏng theo tài liệu
    """
    r = np.linalg.norm(x)
    if r == 0:
        return 0
    theta = np.arctan2(x[1], x[0])
    return r + 0.8 * np.sin(4 * theta) * r

def run_demo(f, history, bounds):
    # Chọn 16 khung hình cách đều nhau
    total_frames = len(history)
    selected_indices = np.linspace(0, total_frames - 1, 16, dtype=int)
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12), constrained_layout=True)
    axes = axes.flatten()
    
    # Lưới nền
    x = np.linspace(bounds[0], bounds[1], 100)
    y = np.linspace(bounds[0], bounds[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f([X[i, j], Y[i, j]])
    
    for idx, frame_idx in enumerate(selected_indices):
        ax = axes[idx]
        data = history[frame_idx]
        mu = data['mu']
        sigma = data['sigma']
        Sigma = data['Sigma']
        samples = data['samples']
        iter_count = data['k'] 
        ax.set_title(f"Iter: {iter_count}", fontsize=10)
        
        ax.contourf(X, Y, Z, levels=30, cmap='viridis')
        
        # Các đường Elip hiệp phương sai
        vals, vecs = np.linalg.eigh(Sigma)
        order = vals.argsort()[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        
        # Góc xoay của elip
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        
        # Các vòng tròn độ lệch chuẩn
        for n_std in range(1, 4):
            width, height = 2 * n_std * sigma * np.sqrt(vals)
            ell = Ellipse(xy=mu, width=width, height=height, 
                          angle=theta, edgecolor='white', facecolor='none', 
                          lw=1)
            ax.add_patch(ell)
            
        # Các mẫu thử nghiệm 
        ax.scatter(samples[:, 0], samples[:, 1], c='white', s=10, alpha=0.8)

        # Thiết lập trục
        ax.set_xlim(bounds)
        ax.set_ylim(bounds)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Nhãn trục
        if idx >= 12:
            ax.set_xlabel('$x_1$')
        if idx % 4 == 0:
            ax.set_ylabel('$x_2$')

    plt.suptitle("Covariance Matrix Adaptation - demo", fontsize=16)
    plt.show()

if __name__ == "__main__":
    np.random.seed(42)
    start_point = [3.0, 3.0] # Điểm bắt đầu lệch tâm
    k_max = 50 # Số vòng lặp tối đa
    
    final_mu, history = covariance_matrix_adaptation(
        flower_function, 
        start_point, 
        k_max, 
        sigma=0.5,
        seed=42
    )
    run_demo(flower_function, history, bounds=[-4, 4])