import numpy as np
import pandas as pd
import time
from tqdm import tqdm

from config import *

def run_experiment():
    all_results = []

    # Thanh tiến trình tổng thể
    total_jobs = len(BENCHMARKS) * len(DIMENSIONS) * len(ALGORITHMS)
    pbar = tqdm(total=total_jobs, desc="Tổng tiến độ")

    for bench_name, bench_class in BENCHMARKS.items():
        for dim in DIMENSIONS:
            # Khởi tạo hàm benchmark
            benchmark_func = bench_class(dim=dim)
            
            for algo_name, algo_info in ALGORITHMS.items():
                algo_class = algo_info["class"]
                
                # Chuẩn bị tham số cho thuật toán
                params = algo_info["params"].copy()
                params["benchmark_func"] = benchmark_func
                params["max_iter"] = MAX_ITER
                
                histories = []

                for i in range(N_RUNS):
                    params["random_state"] = i 
                    
                    optimizer = algo_class(**params)

                    start_time = time.perf_counter()
                    best_solution, best_fitness = optimizer.solve()
                    end_time = time.perf_counter()
                    
                    execution_time = end_time - start_time
                    convergence = optimizer.convergence_history
                    
                    # Đảm bảo convergence có độ dài MAX_ITER
                    if len(convergence) < MAX_ITER:
                        last_val = convergence[-1] if convergence else np.inf
                        convergence = np.pad(convergence, (0, MAX_ITER - len(convergence)), 
                                             'constant', constant_values=last_val)
                    
                    histories.append(convergence[:MAX_ITER])

                    run_result = {
                        "benchmark": bench_name,
                        "dim": dim,
                        "algorithm": algo_name,
                        "run": i,
                        "best_fitness": best_fitness,
                        "time": execution_time
                    }
                    all_results.append(run_result)

                pbar.update(1)
                
                # Chỉ lưu lịch sử hội tụ cho dimension đầu tiên (convergence và robustness)
                if dim == DIMENSIONS[0]:
                    # Tính trung bình lịch sử hội tụ và lưu
                    avg_history = np.mean(np.array(histories), axis=0)
                    for iter_k, fit_val in enumerate(avg_history):
                        history_result = {
                            "benchmark": bench_name,
                            "dim": dim,
                            "algorithm": algo_name,
                            "run": "avg_history",
                            "iteration": iter_k,
                            "best_fitness": fit_val
                        }
                        all_results.append(history_result)

    pbar.close()
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RESULTS_FILE, index=False)
    print(f"Đã lưu kết quả thực nghiệm vào {RESULTS_FILE}")

if __name__ == "__main__":
    run_experiment()