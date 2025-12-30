import os
import random
from modules.input_generator import HashiGenerator

def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def gen_inputs(number_of_testcase, input_dir='inputs'):
    create_dir(input_dir)
    print("Generating benchmark inputs...")

    configs = [
        (7, 7, (6, 8)),
        (9, 9, (10, 12)),
        (11, 11, (14, 16)),
        (13, 13, (18, 22)),
        (17, 17, (25, 30)),
        (20, 20, (40, 50)),
    ]

    generated_files = []
    
    for w, h, (min_isl, max_isl) in configs:
        for i in range(1, number_of_testcase + 1):
            num_islands = random.randint(min_isl, max_isl)
            gen = HashiGenerator(width=w, height=h, num_islands=num_islands)
            grid = gen.generate()
            
            filename = f"testcase_{w}x{h}_{i}.txt"
            filepath = os.path.join(input_dir, filename)
            
            try:
                gen.save_to_file(filepath)
                generated_files.append(filepath)
                print(f"  [OK] Generated {filename} (Size: {w}x{h}, Islands: ~{len(gen.islands)})")
            except IOError as e:
                print(f"  [ERR] Could not write to {filename}: {e}")
                
    print(f"\n>> Successfully created {len(generated_files)} input files in '{input_dir}'.")
    return generated_files

def read_grid_from_file(path):
    grid = []
    with open(path, "r") as f:
        for line in f:
            row = [int(x.strip()) for x in line.split(",")]
            grid.append(row)
    return grid

def format_solution(solver, model):
    rows, cols = solver.rows, solver.cols
    # Initialize output grid with "0" strings
    out_grid = [["0" for _ in range(cols)] for _ in range(rows)]
    
    # Fill island numbers
    for isl in solver.islands:
        out_grid[isl['r']][isl['c']] = str(isl['val'])
        
    # Fill bridges
    model_set = set(model)
    for e in solver.potential_edges:
        v1 = solver.var_map[(e['idx'], 1)]
        v2 = solver.var_map[(e['idx'], 2)]
        
        count = 0
        if v2 in model_set: count = 2
        elif v1 in model_set: count = 1
        
        if count > 0:
            # Choose character based on style
            # Horizontal: 1='-', 2='='
            # Vertical: 1='|', 2='$'
            char = ""
            if e['type'] == 'h':
                char = "=" if count == 2 else "-"
                for c in range(e['c_min'] + 1, e['c_max']):
                    out_grid[e['r']][c] = char
            else: # Vertical
                char = "$" if count == 2 else "|"
                for r in range(e['r_min'] + 1, e['r_max']):
                    out_grid[r][e['c']] = char
    return out_grid
