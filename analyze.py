import csv
import matplotlib.pyplot as plt
import os
import math
from config import TIMEOUT

def analyze_results():
    csv_file = os.path.join('output', 'benchmark.csv')
    if not os.path.exists(csv_file):
        print(f"File {csv_file} not found. Run benchmark.py first.")
        return

    data = {}
    sizes = []

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            algo = row['algorithm']
            size = int(row['size'])
            time_str = row['avg_time']
            
            if size not in sizes:
                sizes.append(size)
            
            if algo not in data:
                data[algo] = {'x': [], 'y': []}
            
            val = 0.0
            if time_str == f'>{TIMEOUT}':
                val = TIMEOUT
            else:
                val = float(time_str)
                if val < 0.0001: 
                    val = 0.0001
                if val > TIMEOUT:
                    val = TIMEOUT
            
            data[algo]['x'].append(size)
            data[algo]['y'].append(val)

    sizes.sort()
    
    plt.figure(figsize=(12, 8))
    
    styles = {
        'pysat': {'marker': 'o', 'color': '#1f77b4', 'label': 'PySAT', 'linestyle': '-'},
        'astar': {'marker': 's', 'color': '#2ca02c', 'label': 'A*', 'linestyle': '--'},
        'backtracking': {'marker': '^', 'color': '#ff7f0e', 'label': 'Backtracking', 'linestyle': '-.'},
        'brute_force': {'marker': 'x', 'color': '#d62728', 'label': 'Brute Force', 'linestyle': ':'}
    }
    
    for algo, coords in data.items():
        points = sorted(zip(coords['x'], coords['y']), key=lambda p: p[0])
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        
        style = styles.get(algo, {'marker': 'o', 'color': 'black', 'label': algo, 'linestyle': '-'})
        
        plt.plot(xs, ys, 
                 marker=style['marker'], 
                 color=style['color'], 
                 label=style['label'],
                 linestyle=style['linestyle'],
                 linewidth=2,
                 markersize=8)
        
        # Đánh dấu các điểm bị Timeout
        for x, y in zip(xs, ys):
            if y >= TIMEOUT:
                 plt.text(x, y * 1.1, "Timeout", ha='center', va='bottom', fontsize=9, color='red', fontweight='bold')

    plt.yscale('log')
    
    plt.title('Performance Comparison: Runtime vs Map Size (Log Scale)', fontsize=16, fontweight='bold')
    plt.xlabel('Map Size (N x N)', fontsize=12)
    plt.ylabel('Average Time (seconds) - Log Scale', fontsize=12)
    
    plt.axhline(y=TIMEOUT, color='r', linestyle='--', alpha=0.5, linewidth=1)
    plt.text(min(sizes), TIMEOUT, f' Timeout Limit ({TIMEOUT}s)', color='red', va='bottom', ha='left')

    plt.xticks(sizes, [f'{s}x{s}' for s in sizes], fontsize=10)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.legend(fontsize=11, loc='upper left')
    plt.tight_layout()
    
    output_plot = os.path.join('output', 'benchmark_plot.png')
    plt.savefig(output_plot, dpi=300)
    print(f"Plot saved to {output_plot}")
    plt.show()

if __name__ == '__main__':
    analyze_results()