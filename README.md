# AI Fundamentals Project 1

Dự án về các thuật toán tối ưu hóa bầy đàn cho môn học Cơ sở Trí tuệ Nhân tạo. Dự án bao gồm triển khai và so sánh các thuật toán tối ưu hóa cho cả bài toán liên tục (Continuous Optimization) và bài toán rời rạc (Discrete Optimization - TSP).

## Cài đặt
```bash
pip install -r requirements.txt
```

Các thư viện sẽ được cài đặt:
- `numpy==1.26.0` - Thư viện tính toán số học
- `matplotlib==3.8.0` - Thư viện vẽ đồ thị
- `seaborn==0.13.0` - Thư viện trực quan hóa dữ liệu
- `imageio>=2.9.0` - Thư viện xử lý ảnh/GIF
- `imageio[ffmpeg]` - Hỗ trợ video/animation

## Cấu trúc thư mục

```
AI-Fundamentals-Proj1/
│
├── LICENSE                          
├── README.md                        # Tài liệu hướng dẫn
├── requirements.txt                 # Danh sách thư viện cần thiết
│
├── src/                            # Source code chính
│   ├── algorithms/                 # Các thuật toán tối ưu hóa
│   │   ├── continuous/            # Thuật toán cho bài toán liên tục
│   │   │   ├── __init__.py
│   │   │   ├── base.py            # Lớp cơ sở cho các thuật toán
│   │   │   ├── hill_climbing.py   # Hill Climbing
│   │   │   ├── simulated_annealing.py  # Simulated Annealing
│   │   │   ├── artificial_bee_colony.py  # Artificial Bee Colony (ABC)
│   │   │   ├── cuckoo_search.py   # Cuckoo Search (CS)
│   │   │   └── firefly_algorithm.py  # Firefly Algorithm (FA)
│   │   │
│   │   └── discrete/              # Thuật toán cho bài toán rời rạc (TSP)
│   │       ├── __init__.py
│   │       ├── tsp_problem.py     # Định nghĩa bài toán TSP
│   │       ├── astar.py           # Thuật toán A* (tìm kiếm có thông tin)
│   │       ├── aco.py             # Ant Colony Optimization (ACO)
│   │       ├── pso.py             # Particle Swarm Optimization (PSO)
│   │       └── genetic_algorithm.py  # Genetic Algorithm (GA)
│   │
│   └── utils/                     # Các tiện ích hỗ trợ
│       ├── __init__.py
│       └── convergence_visualizer.py  # Trực quan hóa quá trình hội tụ
│
├── experiments/                    # Các thí nghiệm và demo
│   ├── continuous/                # Thí nghiệm cho bài toán liên tục
│   │   ├── __init__.py
│   │   ├── config.py              # Cấu hình thí nghiệm
│   │   ├── benchmarks.py          # Các hàm benchmark (Sphere, Rastrigin)
│   │   ├── algo_visualization.py  # Trực quan hóa thuật toán (2D/3D)
│   │   ├── comparison.py          # So sánh các thuật toán
│   │   ├── params_analysis.py     # Phân tích độ nhạy tham số
│   │   └── draw_plots.py          # Vẽ các biểu đồ phân tích
│   │
│   └── discrete/                  # Thí nghiệm cho bài toán TSP
│       ├── __init__.py
│       ├── demo_functions.py      # Các hàm demo chính
│       ├── run_experiments.py     # Chạy thí nghiệm so sánh
│       ├── tsp_comparator.py      # So sánh các thuật toán TSP
│       ├── visualizer.py          # Trực quan hóa kết quả TSP
│       └── algorithm_result.py    # Lớp lưu trữ kết quả
│
├── results/                        # Thư mục chứa kết quả (được tạo khi chạy)
├── plots/                          # Thư mục chứa các biểu đồ (được tạo khi chạy)
└── visualizations/                 # Thư mục chứa animation/GIF (được tạo khi chạy)
```

## Hướng dẫn sử dụng

### Continuous Optimization

#### 1. Trực quan hóa thuật toán (Animation 2D/3D)

Chạy file `algo_visualization.py` để xem hoạt động của các thuật toán:

```bash
cd experiments/continuous
python algo_visualization.py
```

**Output (thư mục `visualizations/`):**
- `*_animation_2d.gif` - Animation 2D
- `*_animation_3d.gif` - Animation 3D
- `*_trajectories_2d.png` - Quỹ đạo các cá thể 2D
- `*_trajectories_3d.png` - Quỹ đạo các cá thể 3D
- `*_convergence_curve_2d.png` - Đường hội tụ

#### 2. So sánh các thuật toán

Chạy thí nghiệm so sánh toàn diện:

```bash
cd experiments/continuous
python comparison.py
```

**Sau đó vẽ biểu đồ phân tích:**

```bash
python draw_plots.py
```

Output trong thư mục `plots/`:
- `convergence_all_benchmarks.png` - Tốc độ hội tụ
- `robustness_all_benchmarks.png` - Độ ổn định (boxplot)
- `complexity_time.png` - Thời gian chạy
- `scalability_dimensions.png` - Khả năng mở rộng theo chiều

#### 3. Phân tích độ nhạy tham số

Chạy phân tích độ nhạy của các tham số:

```bash
cd experiments/continuous
python params_analysis.py
```

Sau đó vẽ biểu đồ:

```bash
python draw_plots.py
```

Output: `plots/sensitivity_*.png` - Biểu đồ phân tích độ nhạy từng tham số

### Discrete Optimization (TSP)

#### 1. Demo so sánh cơ bản

Chạy so sánh cơ bản các thuật toán trên TSP:

```bash
cd experiments/discrete
python run_experiments.py
```

Hoặc sử dụng các demo function riêng lẻ:

```python
from demo_functions import (
    demo_basic_comparison,
    demo_robustness_analysis,
    demo_scalability_analysis,
    demo_parameter_sensitivity,
    demo_comprehensive_analysis
)

# Chạy demo cơ bản
demo_basic_comparison()

# Phân tích độ ổn định
demo_robustness_analysis()

# Phân tích khả năng mở rộng
demo_scalability_analysis()

# Phân tích độ nhạy tham số
demo_parameter_sensitivity()

# Phân tích toàn diện
demo_comprehensive_analysis()
```

#### 2. Sử dụng trực tiếp các thuật toán

**Ví dụ với ACO:**

```python
from src.algorithms.discrete.tsp_problem import create_tsp
from src.algorithms.discrete.aco import ACO

# Tạo bài toán TSP với 30 thành phố
tsp = create_tsp(n_cities=30, symmetric=True, seed=42)

# Khởi tạo ACO
aco = ACO(
    tsp_problem=tsp,
    n_ants=30,
    n_iterations=100,
    alpha=2.0,
    beta=5.0,
    rho=0.5,
    Q=100.0
)

# Giải bài toán
best_tour, best_cost, cost_history, elapsed_time = aco.solve()

print(f"Best tour: {best_tour}")
print(f"Best cost: {best_cost:.2f}")
print(f"Time: {elapsed_time:.2f}s")

# Vẽ lộ trình
aco.plot_tour(title="ACO Solution")

# Vẽ đường hội tụ
aco.plot_convergence()
```

**Ví dụ với PSO:**

```python
from src.algorithms.discrete.pso import PSO

pso = PSO(
    tsp_problem=tsp,
    n_particles=30,
    n_iterations=100,
    w=0.7,
    c1=0.9,
    c2=0.3,
    topology='gbest',  # hoặc 'lbest'
    seed=42
)

best_tour, best_cost, cost_history, elapsed_time = pso.solve()
```

**Ví dụ với Genetic Algorithm:**

```python
from src.algorithms.discrete.genetic_algorithm import GeneticAlgorithm

ga = GeneticAlgorithm(
    tsp_problem=tsp,
    population_size=100,
    n_generations=100,
    crossover_rate=0.8,
    mutation_rate=0.02,
    tournament_size=5,
    elitism_size=2,
    seed=42
)

best_tour, best_cost, cost_history, elapsed_time = ga.solve()
```

#### 3. Sử dụng TSPComparator để so sánh

```python
from experiments.discrete.tsp_comparator import TSPComparator
from experiments.discrete.visualizer import Visualizer
from src.algorithms.discrete.tsp_problem import create_tsp

# Tạo bài toán
tsp = create_tsp(n_cities=50, symmetric=True, seed=42)

# So sánh các thuật toán
comparator = TSPComparator(seed=42)
results = comparator.compare_algorithms(tsp)

# Trực quan hóa kết quả
visualizer = Visualizer()
visualizer.create_summary_table(results)
visualizer.plot_convergence_comparison(results)
visualizer.plot_performance_comparison(results)
```

## Tác giả

* Đinh Đức Anh Khoa - 23122001
* Nguyễn Lê Hoàng Trung - 23122004
* Đinh Đức Tài - 23122013