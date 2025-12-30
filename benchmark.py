import os
import time
import csv
import multiprocessing
import traceback
from datetime import datetime

from utils.helper import *
from config import *

def run_solver_worker(algo_name, grid, return_dict):
    try:
        solver_class = ALGORITHMS[algo_name]
        solver = solver_class(grid)
        
        start_time = time.time()
        result = solver.solve()
        end_time = time.time()
        
        elapsed = end_time - start_time
        
        return_dict['result'] = result
        return_dict['time'] = elapsed
        
        solver_data = {
            'rows': solver.rows,
            'cols': solver.cols,
            'islands': solver.islands,
            'potential_edges': solver.potential_edges,
            'var_map': solver.var_map
        }
        return_dict['solver_data'] = solver_data
        
    except Exception as e:
        return_dict['error'] = str(e)
        return_dict['traceback'] = traceback.format_exc()

class SolverDataWrapper:
    def __init__(self, data):
        self.__dict__.update(data)

def run_benchmark():
    # Generate inputs
    input_dir = 'inputs'
    gen_inputs(NUM_TESTCASE, input_dir)
    
    # Setup output
    output_dir = 'output'
    solutions_dir = os.path.join(output_dir, 'solutions')
    create_dir(output_dir)
    create_dir(solutions_dir)
    
    csv_file = os.path.join(output_dir, 'benchmark.csv')
    
    # Prepare CSV
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['size', 'algorithm', 'avg_time'])
    
    manager = multiprocessing.Manager()
    
    for size in TEST_SIZES:
        print(f"\nProcessing Map Size: {size}x{size}")
        
        for algo_name in ALGORITHMS.keys():
            if algo_name == 'brute_force' and size >= 9:
                print(f"  Skipping {algo_name} for size {size}x{size} (Predictable Timeout)")
                with open(csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([size, algo_name, f">{TIMEOUT}"])
                
                output_filename = f"{algo_name}_{size}x{size}.txt"
                output_path = os.path.join(solutions_dir, output_filename)
                with open(output_path, 'w') as f:
                    f.write("Skipped: Brute Force disabled for size >= 9x9.")
                continue

            total_time = 0
            success_count = 0
            timeout_occurred = False
            
            # Prepare solution file content
            solution_output_content = []
            
            print(f"  Running {algo_name}...")
            for i in range(1, NUM_TESTCASE + 1):
                input_file = f"testcase_{size}x{size}_{i}.txt"
                input_path = os.path.join(input_dir, input_file)
                
                if not os.path.exists(input_path):
                    print(f"    Warning: {input_file} not found.")
                    continue
                
                grid = read_grid_from_file(input_path)
                
                # Shared dict for results
                return_dict = manager.dict()
                
                # Start process
                p = multiprocessing.Process(target=run_solver_worker, args=(algo_name, grid, return_dict))
                p.start()
                p.join(TIMEOUT)
                
                result_time = 0
                result_model = None
                solver_data = None
                
                if p.is_alive():
                    print(f"    {input_file}: Timeout (> {TIMEOUT}s)")
                    p.terminate()
                    p.join()
                    timeout_occurred = True
                    result_time = TIMEOUT
                else:
                    if 'error' in return_dict:
                        print(f"    {input_file}: Error - {return_dict['error']}")
                    else:
                        result_time = return_dict.get('time', 0)
                        result_model = return_dict.get('result')
                        solver_data = return_dict.get('solver_data')
                        print(f"    {input_file}: {result_time:.4f}s")
                        success_count += 1
                
                total_time += result_time
                
                # Format solution
                sol_str = ""
                if result_model:
                    solver_wrapper = SolverDataWrapper(solver_data)
                    formatted_grid = format_solution(solver_wrapper, result_model)
                    
                    sol_lines = [f"Solution for {input_file}:"]
                    for row in formatted_grid:
                        sol_lines.append(str(row))
                    sol_str = "\n".join(sol_lines)
                else:
                    sol_str = f"Solution for {input_file}:\nNo solution found or Timed out."
                
                solution_output_content.append(sol_str)
            
            # Calculate Average
            avg_time = total_time / 5
            avg_str = f"{avg_time:.4f}"
            if timeout_occurred or avg_time >= TIMEOUT:
                avg_str = f">{TIMEOUT}"
            
            # Write to CSV
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([size, algo_name, avg_str])
            
            # Write solutions to file
            output_filename = f"{algo_name}_{size}x{size}.txt"
            output_path = os.path.join(solutions_dir, output_filename)
            with open(output_path, 'w') as f:
                f.write("\n\n".join(solution_output_content))
                f.write("\n")

if __name__ == '__main__':
    run_benchmark()
