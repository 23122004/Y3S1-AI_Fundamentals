# analysis/run_analysis.py
import pandas as pd
import numpy as np
from tqdm import tqdm

from config import *

def run_experiment():
    """
    Chạy toàn bộ thí nghiệm phân tích độ nhạy và lưu kết quả.
    """
    all_results = {}
    benchmark = BENCHMARKS["Rastrigin"](dim=DEFAULT_DIMENSIONS)

    for algo_name, config in EXPERIMENTS.items():
        print(f"--- Đang xử lý: {algo_name} ---")
        algo_class = config['class']
        defaults = config['defaults']
        params_to_test = config['params_to_test']

        algo_results = {}

        # Lặp qua từng tham số
        for param_name, param_values in params_to_test.items():
            print(f"    Phân tích tham số: {param_name}")
            
            param_results = {}
            for value in tqdm(param_values, desc=f"    {param_name}", leave=False):
                # Tạo bản sao các tham số mặc định và cập nhật tham số đang kiểm tra
                current_params = defaults.copy()
                current_params[param_name] = value
                
                run_fitness_values = []
                
                # Chạy N_RUNS lần để lấy kết quả thống kê
                for i in range(N_RUNS):
                    algo = algo_class(
                        benchmark_func=benchmark,
                        max_iter=MAX_ITER,
                        random_state=i,
                        **current_params
                    )
                    algo.solve()
                    run_fitness_values.append(algo.best_fitness)
                param_results[value] = run_fitness_values
            algo_results[param_name] = param_results
        all_results[algo_name] = algo_results

    rows = []
    for algo_name, algo_data in all_results.items():
        for param_name, param_data in algo_data.items():
            for param_value, fitness_values in param_data.items():
                for run_idx, fitness in enumerate(fitness_values):
                    rows.append({
                        'Algorithm': algo_name,
                        'Parameter': param_name,
                        'Value': param_value,
                        'Run': run_idx + 1,
                        'Fitness': fitness
                    })
    
    df = pd.DataFrame(rows)
    
    df.to_csv(SENSITIVITY_RESULTS_FILE, index=False, encoding='utf-8')
    
    print(f"Kết quả đã được lưu vào: {SENSITIVITY_RESULTS_FILE}")
    print(f"Tổng số dòng: {len(df)}")

if __name__ == "__main__":
    run_experiment()