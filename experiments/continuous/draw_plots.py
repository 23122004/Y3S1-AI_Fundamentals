import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from config import *
sns.set_theme(style="whitegrid", palette="muted")

def plot_convergence_all_benchmarks(df, dim):
    benchmarks = list(BENCHMARKS.keys())
    n_benchmarks = len(benchmarks)
    
    fig, axes = plt.subplots(1, n_benchmarks, figsize=(12 * n_benchmarks, 7))
    if n_benchmarks == 1:
        axes = [axes]
    
    for idx, bench_name in enumerate(benchmarks):
        # Lọc dữ liệu lịch sử hội tụ
        history_df = df[
            (df["run"] == "avg_history") &
            (df["benchmark"] == bench_name) &
            (df["dim"] == dim)
        ]
        
        ax = axes[idx]
        sns.lineplot(
            data=history_df,
            x="iteration",
            y="best_fitness",
            hue="algorithm",
            lw=2,
            ax=ax
        )
        ax.set_title(f"Convergence Speed (Iterations) - {bench_name}", fontsize=14)
        ax.set_xlabel("Iteration", fontsize=12)
        ax.set_ylabel("Avg Best Fitness", fontsize=12)
        ax.legend(title="Algorithms")
    
    plt.tight_layout()
    filename = PLOTS_DIR / "convergence_all_benchmarks.png"
    plt.savefig(filename)
    plt.close()
    print(f"Saved: {filename}")

def plot_robustness_all_benchmarks(df, dim):
    benchmarks = list(BENCHMARKS.keys())
    n_benchmarks = len(benchmarks)
    
    fig, axes = plt.subplots(1, n_benchmarks, figsize=(12 * n_benchmarks, 7))
    if n_benchmarks == 1:
        axes = [axes]
    
    for idx, bench_name in enumerate(benchmarks):
        # Lọc dữ liệu kết quả cuối cùng (không phải history)
        run_df = df[
            (df["run"] != "avg_history") &
            (df["benchmark"] == bench_name) &
            (df["dim"] == dim)
        ]
        
        ax = axes[idx]
        sns.boxplot(
            data=run_df,
            x="algorithm",
            y="best_fitness",
            palette="Set2",
            ax=ax
        )
        
        ax.set_title(f"Algorithm Robustness - {bench_name}", fontsize=14)
        ax.set_xlabel("Algorithm", fontsize=12)
        ax.set_ylabel("Final Fitness (Log Scale)", fontsize=12)
        ax.set_yscale('log')
        ax.tick_params(axis='x')
    
    plt.tight_layout()
    filename = PLOTS_DIR / "robustness_all_benchmarks.png"
    plt.savefig(filename)
    plt.close()
    print(f"Saved: {filename}")

def plot_scalability_all_benchmarks(df):
    benchmarks = list(BENCHMARKS.keys())
    n_benchmarks = len(benchmarks)
    
    # 1. Scalability - Fitness
    fig, axes = plt.subplots(1, n_benchmarks, figsize=(12 * n_benchmarks, 7))
    if n_benchmarks == 1:
        axes = [axes]
    
    for idx, bench_name in enumerate(benchmarks):
        run_df = df[
            (df["run"] != "avg_history") &
            (df["benchmark"] == bench_name)
        ]
        avg_fit_df = run_df.groupby(["algorithm", "dim"])["best_fitness"].mean().reset_index()

        ax = axes[idx]
        sns.lineplot(
            data=avg_fit_df,
            x="dim",
            y="best_fitness",
            hue="algorithm",
            style="algorithm",
            markers=True,
            dashes=False,
            lw=2,
            markersize=8,
            ax=ax
        )

        ax.set_title(f"Scalability (Fitness) - {bench_name}", fontsize=14)
        ax.set_xlabel("Dimension", fontsize=12)
        ax.set_ylabel("Avg Best Fitness", fontsize=12)
        ax.set(xticks=DIMENSIONS)
        ax.legend(title="Algorithms")
    
    plt.tight_layout()
    filename = PLOTS_DIR / "scalability_fitness_all_benchmarks.png"
    plt.savefig(filename)
    plt.close()
    print(f"Saved: {filename}")
    
    # 2. Scalability - Time
    fig, axes = plt.subplots(1, n_benchmarks, figsize=(12 * n_benchmarks, 7))
    if n_benchmarks == 1:
        axes = [axes]
    
    for idx, bench_name in enumerate(benchmarks):
        # Lọc và tính thời gian trung bình theo dim
        run_df = df[
            (df["run"] != "avg_history") &
            (df["benchmark"] == bench_name)
        ]
        avg_time_df = run_df.groupby(["algorithm", "dim"])["time"].mean().reset_index()

        ax = axes[idx]
        sns.lineplot(
            data=avg_time_df,
            x="dim",
            y="time",
            hue="algorithm",
            style="algorithm",
            markers=True,
            dashes=False,
            lw=2,
            markersize=8,
            ax=ax
        )

        ax.set_title(f"Scalability (Time) - {bench_name}", fontsize=14)
        ax.set_xlabel("Dimension", fontsize=12)
        ax.set_ylabel("Avg Execution Time (Seconds)", fontsize=12)
        ax.set(xticks=DIMENSIONS)
        ax.legend(title="Algorithms")
    
    plt.tight_layout()
    filename = PLOTS_DIR / "scalability_time_all_benchmarks.png"
    plt.savefig(filename)
    plt.close()
    print(f"Saved: {filename}")

def plot_param_sensitivity():
    # Đọc dữ liệu từ file CSV
    try:
        df = pd.read_csv(SENSITIVITY_RESULTS_FILE)
    except FileNotFoundError:
        print(f"Error: File {SENSITIVITY_RESULTS_FILE} not found. Please run 'paramsAnalysis.py' first.")
        return
    
    sns.set_theme(style="whitegrid")

    # Lấy danh sách các thuật toán và tham số duy nhất
    algorithms = df['Algorithm'].unique()
    
    for algo_name in algorithms:
        algo_df = df[df['Algorithm'] == algo_name]
        parameters = algo_df['Parameter'].unique()
        
        # Tính số lượng tham số để xác định layout subplots
        n_params = len(parameters)
        if n_params == 0:
            continue
            
        # Tính số hàng và cột cho subplots
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols  
        
        # Tạo figure với subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        fig.suptitle(f"Parameter Sensitivity Analysis - {algo_name} - Rastrigin", fontsize=18, y=0.995)
        
        if n_params == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
        
        for idx, param_name in enumerate(parameters):
            param_df = algo_df[algo_df['Parameter'] == param_name]
            param_df['Value_str'] = param_df['Value'].astype(str)
            
            ax = axes[idx]
            sns.boxplot(x='Value_str', y='Fitness', data=param_df, palette="colorblind", ax=ax)
            
            # tiêu đề và nhãn
            ax.set_title(f"Parameter: {param_name}", fontsize=14)
            ax.set_xlabel(f"{param_name}", fontsize=11)
            ax.set_ylabel("Best Fitness (small is better)", fontsize=11)
            ax.tick_params(axis='x')
        for idx in range(n_params, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        # Lưu biểu đồ
        plot_filename = PLOTS_DIR / f"sensitivity_{algo_name}.png"
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved sensitivity plot for {algo_name}")

def algo_comparison():
    if not RESULTS_FILE.exists():
        print(f"File {RESULTS_FILE} not found. Please run run_experiment.py first.")
        sys.exit(1)
    df = pd.read_csv(RESULTS_FILE)
    
    # Scalability plots (2 biểu đồ cho 2 benchmarks)
    print("Creating scalability plots...")
    plot_scalability_all_benchmarks(df)
    
    # Convergence và Robustness chỉ với dimension đầu tiên (2 biểu đồ cho 2 benchmarks)
    first_dim = DIMENSIONS[0]
    print(f"Creating convergence and robustness plots for D={first_dim}...")
    plot_convergence_all_benchmarks(df, first_dim)
    plot_robustness_all_benchmarks(df, first_dim)
            
if __name__ == "__main__":
    plot_param_sensitivity()
    algo_comparison()
    print(f"All plots saved to directory {PLOTS_DIR}")