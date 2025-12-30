"""
Genetic Algorithm (GA) for Traveling Salesman Problem (TSP)

Thuật toán di truyền để giải bài toán TSP rời rạc.
Sử dụng biểu diễn nhiễm sắc thể dạng hoán vị (permutation encoding).

Genetic Algorithm (GA) dựa trên học thuyết Darwin về "sự sinh tồn của những cá thể 
khỏe mạnh nhất". Thuật toán mô phỏng quá trình tiến hóa tự nhiên để tìm kiếm 
giải pháp tối ưu cho bài toán TSP.

Tham khảo:
- Holland, J. H. (1992). Adaptation in natural and artificial systems.
- Razali, N. M., & Geraghty, J. (2011). Genetic algorithm performance with different selection strategies in solving TSP.
"""

import numpy as np
import random
import time
from typing import List, Tuple, Optional
try:
    from .tsp_problem import TSPProblem, create_tsp
except ImportError:
    from tsp_problem import TSPProblem, create_tsp

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Plotting functions will be disabled.")


class GeneticAlgorithm:
    """
    Thuật toán di truyền (Genetic Algorithm) cho bài toán TSP.
    
    Attributes:
        tsp: Đối tượng bài toán TSP
        n_cities: Số lượng thành phố
        population_size: Kích thước quần thể (N)
        n_generations: Số thế hệ tối đa (generation_max)
        p_crossover: Xác suất lai ghép (0 < p_crossover <= 1)
        p_mutation: Xác suất đột biến (0 < p_mutation <= 1)
        tournament_k: Số cá thể tham gia tournament selection
        epsilon: Ngưỡng cải thiện để dừng sớm
        patience: Số thế hệ không cải thiện trước khi dừng
        goal_fitness: Mục tiêu fitness để dừng
        
        population: Quần thể hiện tại (danh sách các nhiễm sắc thể)
        fitness_values: Giá trị fitness của từng cá thể
        best_chromosome: Nhiễm sắc thể tốt nhất
        best_fitness: Fitness tốt nhất
        best_cost: Chi phí tốt nhất (1/fitness)
    """
    
    def __init__(
        self,
        tsp_problem: TSPProblem,
        population_size: int = 100,
        n_generations: int = 500,
        p_crossover: float = 0.8,
        p_mutation: float = 0.2,
        tournament_k: int = 5,
        epsilon: float = 1e-6,
        patience: int = 50,
        goal_fitness: Optional[float] = None,
        seed: Optional[int] = None
    ):
        """
        Khởi tạo Genetic Algorithm solver.
        
        Args:
            tsp_problem: Bài toán TSP cần giải
            population_size: Kích thước quần thể N (>= 2)
            n_generations: Số thế hệ tối đa (>= 1)
            p_crossover: Xác suất lai ghép (0 < p <= 1)
            p_mutation: Xác suất đột biến (0 < p <= 1)
            tournament_k: Số cá thể trong tournament (k >= 2)
            epsilon: Ngưỡng cải thiện để dừng sớm
            patience: Số thế hệ không cải thiện trước khi dừng
            goal_fitness: Mục tiêu fitness (dừng khi đạt được)
            seed: Random seed
        """
        # Kiểm tra tham số
        if population_size < 2:
            raise ValueError("Population size must be >= 2")
        if n_generations < 1:
            raise ValueError("Number of generations must be >= 1")
        if not 0 < p_crossover <= 1:
            raise ValueError("Crossover probability must be in (0, 1]")
        if not 0 < p_mutation <= 1:
            raise ValueError("Mutation probability must be in (0, 1]")
        if tournament_k < 2:
            raise ValueError("Tournament k must be >= 2 (need at least 2 parents)")
        if tournament_k > population_size:
            raise ValueError("Tournament k cannot exceed population size")
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        self.tsp = tsp_problem
        self.n_cities = tsp_problem.n_cities
        self.population_size = population_size
        self.n_generations = n_generations
        self.p_crossover = p_crossover
        self.p_mutation = p_mutation
        self.tournament_k = tournament_k
        self.epsilon = epsilon
        self.patience = patience
        self.goal_fitness = goal_fitness
        
        # Khởi tạo quần thể
        self.population = []
        self.fitness_values = []
        self.best_chromosome = None
        self.best_fitness = -float('inf')
        self.best_cost = float('inf')
        self.fitness_history = []
        self.cost_history = []
    
    def _calculate_fitness(self, chromosome: List[int]) -> float:
        """
        Tính fitness của một nhiễm sắc thể.
        
        Hàm fitness: fitness(X) = 1 / (d(xm, x1) + Σ d(xi, xi+1))
        
        Args:
            chromosome: Nhiễm sắc thể (hoán vị các thành phố)
        
        Returns:
            Giá trị fitness (càng cao càng tốt)
        """
        cost = self.tsp.calculate_tour_cost(chromosome)
        if cost == float('inf') or cost == 0:
            return 0.0
        return 1.0 / cost
    
    def _initialize_population(self):
        """
        Khởi tạo quần thể ban đầu với N cá thể ngẫu nhiên.
        Mỗi cá thể là một hoán vị ngẫu nhiên của các thành phố.
        """
        self.population = []
        self.fitness_values = []
        
        for _ in range(self.population_size):
            chromosome = self.tsp.generate_random_tour()
            fitness = self._calculate_fitness(chromosome)
            
            self.population.append(chromosome)
            self.fitness_values.append(fitness)
            
            # Cập nhật best nếu cần
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_chromosome = chromosome.copy()
                self.best_cost = self.tsp.calculate_tour_cost(chromosome)
    
    def _tournament_selection(self) -> Tuple[List[int], List[int]]:
        """
        Lựa chọn cha mẹ bằng Tournament Selection.
        
        Quy trình:
        1. Chọn ngẫu nhiên k cá thể từ quần thể
        2. Sắp xếp k cá thể theo fitness giảm dần
        3. Chọn 2 cá thể tốt nhất làm cha mẹ (rho = 2 cố định)
        
        Returns:
            Tuple của 2 cha mẹ (parent1, parent2)
        """
        # Chọn k cá thể ngẫu nhiên
        indices = random.sample(range(self.population_size), self.tournament_k)
        
        # Lấy fitness của k cá thể
        tournament_fitness = [(i, self.fitness_values[i]) for i in indices]
        
        # Sắp xếp theo fitness giảm dần
        tournament_fitness.sort(key=lambda x: x[1], reverse=True)
        
        # Chọn 2 cá thể tốt nhất (rho = 2 cố định)
        parent1_idx = tournament_fitness[0][0]
        parent2_idx = tournament_fitness[1][0]
        
        return self.population[parent1_idx].copy(), self.population[parent2_idx].copy()
    
    def _order_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """
        Lai ghép Order Crossover (OX).
        
        Quy trình:
        1. Chọn ngẫu nhiên một đoạn con từ parent1
        2. Sao chép đoạn con đó vào child ở cùng vị trí
        3. Điền các vị trí còn trống bằng các thành phố từ parent2,
           theo thứ tự xuất hiện trong parent2, bỏ qua các thành phố đã có
        
        Args:
            parent1: Cha mẹ thứ nhất
            parent2: Cha mẹ thứ hai
        
        Returns:
            Con (child chromosome)
        """
        size = len(parent1)
        
        # Chọn 2 điểm cắt ngẫu nhiên
        cut1, cut2 = sorted(random.sample(range(size), 2))
        
        # Khởi tạo child với giá trị None
        child = [None] * size
        
        # Sao chép đoạn từ parent1 vào child
        child[cut1:cut2] = parent1[cut1:cut2]
        
        # Tạo danh sách các thành phố cần điền từ parent2
        # theo thứ tự xuất hiện trong parent2
        remaining = [city for city in parent2 if city not in child]
        
        # Điền vào các vị trí còn trống
        remaining_idx = 0
        for i in range(size):
            if child[i] is None:
                child[i] = remaining[remaining_idx]
                remaining_idx += 1
        
        return child
    
    def _swap_mutation(self, chromosome: List[int]) -> List[int]:
        """
        Đột biến hoán vị (Swap Mutation).
        
        Chọn ngẫu nhiên 2 vị trí và đổi chỗ.
        
        Args:
            chromosome: Nhiễm sắc thể cần đột biến
        
        Returns:
            Nhiễm sắc thể sau khi đột biến
        """
        mutated = chromosome.copy()
        idx1, idx2 = random.sample(range(len(mutated)), 2)
        mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
        return mutated
    
    def _reverse_mutation(self, chromosome: List[int]) -> List[int]:
        """
        Đột biến đảo ngược (Reverse Mutation).
        
        Chọn một đoạn con ngẫu nhiên và đảo ngược thứ tự.
        
        Args:
            chromosome: Nhiễm sắc thể cần đột biến
        
        Returns:
            Nhiễm sắc thể sau khi đột biến
        """
        mutated = chromosome.copy()
        idx1, idx2 = sorted(random.sample(range(len(mutated)), 2))
        mutated[idx1:idx2+1] = reversed(mutated[idx1:idx2+1])
        return mutated
    
    def _mutate(self, chromosome: List[int]) -> List[int]:
        """
        Áp dụng đột biến ngẫu nhiên (swap hoặc reverse).
        
        Args:
            chromosome: Nhiễm sắc thể cần đột biến
        
        Returns:
            Nhiễm sắc thể sau khi đột biến
        """
        if random.random() < 0.5:
            return self._swap_mutation(chromosome)
        else:
            return self._reverse_mutation(chromosome)
    
    def solve(self) -> Tuple[List[int], float, List[float], List[float], float]:
        """
        Thực thi thuật toán di truyền để tìm giải pháp cho TSP.
        
        Returns:
            Tuple chứa:
            - best_tour: Lộ trình tốt nhất
            - best_cost: Chi phí tốt nhất
            - cost_history: Lịch sử chi phí tốt nhất mỗi thế hệ
            - fitness_history: Lịch sử fitness tốt nhất mỗi thế hệ
            - elapsed_time: Thời gian thực hiện (giây)
        """
        print("=" * 70)
        print("GENETIC ALGORITHM FOR TSP")
        print("=" * 70)
        print(f"Population size: {self.population_size}")
        print(f"Max generations: {self.n_generations}")
        print(f"Crossover probability: {self.p_crossover}")
        print(f"Mutation probability: {self.p_mutation}")
        print(f"Tournament k: {self.tournament_k} (select top 2 as parents)")
        print("=" * 70)
        
        start_time = time.time()
        
        # Khởi tạo quần thể
        self._initialize_population()
        self.fitness_history.append(self.best_fitness)
        self.cost_history.append(self.best_cost)
        
        print(f"Generation 0: Best cost = {self.best_cost:.2f}")
        
        # Biến để theo dõi early stopping
        generations_without_improvement = 0
        prev_best_fitness = self.best_fitness
        
        # Vòng lặp tiến hóa
        for generation in range(1, self.n_generations + 1):
            new_population = []
            new_fitness_values = []
            
            # Tạo quần thể mới
            while len(new_population) < self.population_size:
                # Lựa chọn
                parent1, parent2 = self._tournament_selection()
                
                # Lai ghép
                if random.random() < self.p_crossover:
                    child = self._order_crossover(parent1, parent2)
                else:
                    # Nếu không lai ghép, chọn một trong hai cha mẹ
                    child = parent1.copy() if random.random() < 0.5 else parent2.copy()
                
                # Đột biến
                if random.random() < self.p_mutation:
                    child = self._mutate(child)
                
                # Tính fitness
                fitness = self._calculate_fitness(child)
                
                new_population.append(child)
                new_fitness_values.append(fitness)
                
                # Cập nhật best nếu cần
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_chromosome = child.copy()
                    self.best_cost = self.tsp.calculate_tour_cost(child)
            
            # Cập nhật quần thể
            self.population = new_population
            self.fitness_values = new_fitness_values
            
            # Lưu lịch sử
            self.fitness_history.append(self.best_fitness)
            self.cost_history.append(self.best_cost)
            
            # In tiến trình mỗi 10% thế hệ
            if generation % max(1, self.n_generations // 10) == 0:
                print(f"Generation {generation}: Best cost = {self.best_cost:.2f}, "
                      f"Best fitness = {self.best_fitness:.6f}")
            
            # Kiểm tra điều kiện dừng
            
            # 1. Đạt mục tiêu fitness
            if self.goal_fitness is not None and self.best_fitness >= self.goal_fitness:
                print(f"\nReached goal fitness at generation {generation}")
                break
            
            # 2. Không cải thiện sau patience thế hệ
            improvement = self.best_fitness - prev_best_fitness
            if improvement < self.epsilon:
                generations_without_improvement += 1
                if generations_without_improvement >= self.patience:
                    print(f"\nEarly stopping at generation {generation} "
                          f"(no improvement for {self.patience} generations)")
                    break
            else:
                generations_without_improvement = 0
            
            prev_best_fitness = self.best_fitness
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print("\n" + "=" * 70)
        print("OPTIMIZATION COMPLETED")
        print("=" * 70)
        print(f"Best tour found: {self.best_chromosome}")
        print(f"Best cost: {self.best_cost:.2f}")
        print(f"Best fitness: {self.best_fitness:.6f}")
        print(f"Total generations: {len(self.cost_history)}")
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        print("=" * 70)
        
        return (
            self.best_chromosome,
            self.best_cost,
            self.cost_history,
            self.fitness_history,
            elapsed_time
        )
    
    def plot_convergence(self):
        """Vẽ biểu đồ hội tụ của thuật toán GA."""
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available. Cannot plot.")
            return
        
        if not self.cost_history:
            print("No history to plot. Run solve() first.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot cost history
        ax1.plot(self.cost_history, 'b-', linewidth=2, label='Best Cost')
        ax1.set_xlabel('Generation', fontsize=12)
        ax1.set_ylabel('Best Cost', fontsize=12)
        ax1.set_title('GA Cost Convergence', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot fitness history
        ax2.plot(self.fitness_history, 'g-', linewidth=2, label='Best Fitness')
        ax2.set_xlabel('Generation', fontsize=12)
        ax2.set_ylabel('Best Fitness', fontsize=12)
        ax2.set_title('GA Fitness Convergence', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add parameter info
        param_text = (f'Pop: {self.population_size}, Gen: {len(self.cost_history)}\n'
                     f'Pc: {self.p_crossover}, Pm: {self.p_mutation}\n'
                     f'Tournament k: {self.tournament_k}')
        
        fig.text(0.5, 0.02, param_text, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.show()
    
    def plot_tour(self, tour: List[int] = None, title: str = None):
        """
        Vẽ lộ trình TSP.
        
        Args:
            tour: Lộ trình cần vẽ. Nếu None, vẽ best_chromosome.
            title: Tiêu đề của biểu đồ. Nếu None, tự động tạo tiêu đề.
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available. Cannot plot.")
            return
        
        if tour is None:
            tour = self.best_chromosome
        
        if tour is None:
            print("No tour to plot. Run solve() first.")
            return
        
        positions = self.tsp.get_city_positions()
        if positions is None:
            print("City positions not available. Cannot plot tour.")
            return
        
        tour_cost = self.tsp.calculate_tour_cost(tour)
        
        if title is None:
            title = f'GA Best Tour (Cost: {tour_cost:.2f})'
        
        plt.figure(figsize=(10, 8))
        
        # Vẽ các thành phố
        plt.scatter(positions[1:, 0], positions[1:, 1], c='blue', s=100, zorder=3, label='Cities')
        
        # Đánh số các thành phố
        for i in range(1, self.tsp.n_cities + 1):
            plt.annotate(str(i), (positions[i, 0], positions[i, 1]),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        # Vẽ lộ trình
        tour_positions = np.array([positions[city] for city in tour])
        tour_positions = np.vstack([tour_positions, tour_positions[0]])
        
        plt.plot(tour_positions[:, 0], tour_positions[:, 1], 'r-', 
                linewidth=2, zorder=2, label=f'Tour (Cost: {tour_cost:.2f})')
        
        # Đánh dấu điểm bắt đầu
        start_pos = positions[tour[0]]
        plt.scatter([start_pos[0]], [start_pos[1]], c='green', s=200, 
                   marker='*', zorder=4, label=f'Start (City {tour[0]})')
        
        plt.xlabel('X Coordinate', fontsize=12)
        plt.ylabel('Y Coordinate', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def demo_ga():
    """Demo thuật toán GA cho TSP."""
    print("=" * 70)
    print("GENETIC ALGORITHM FOR TSP - DEMO")
    print("=" * 70)
    
    # 1. Tạo bài toán TSP
    N_CITIES = 100
    tsp = create_tsp(n_cities=N_CITIES, symmetric=True, seed=42)
    print(f"\nCreated TSP with {N_CITIES} cities.")
    
    # 2. Định nghĩa các siêu tham số cho GA
    POPULATION_SIZE = 100      # Kích thước quần thể (N >= 2)
    N_GENERATIONS = 2000        # Số thế hệ tối đa (>= 1)
    P_CROSSOVER = 0.8          # Xác suất lai ghép (0 < p <= 1)
    P_MUTATION = 0.2           # Xác suất đột biến (0 < p <= 1)
    TOURNAMENT_K = 10          # Số cá thể trong tournament (k >= 2, chọn top 2 làm cha mẹ)
    EPSILON = 1e-6             # Ngưỡng cải thiện
    PATIENCE = 50              # Số thế hệ không cải thiện trước khi dừng
    
    # 3. Khởi tạo và chạy GA
    ga_solver = GeneticAlgorithm(
        tsp_problem=tsp,
        population_size=POPULATION_SIZE,
        n_generations=N_GENERATIONS,
        p_crossover=P_CROSSOVER,
        p_mutation=P_MUTATION,
        tournament_k=TOURNAMENT_K,
        epsilon=EPSILON,
        patience=PATIENCE,
        seed=42
    )
    
    best_tour, best_cost, cost_history, fitness_history, elapsed_time = ga_solver.solve()
    
    # 4. So sánh với lộ trình ngẫu nhiên
    print("\n" + "=" * 70)
    print("COMPARISON WITH RANDOM TOUR")
    print("=" * 70)
    
    random_tour = tsp.generate_random_tour()
    random_cost = tsp.calculate_tour_cost(random_tour)
    
    print(f"\nRandom tour cost: {random_cost:.2f}")
    print(f"GA best cost: {best_cost:.2f}")
    print(f"Improvement: {((random_cost - best_cost) / random_cost * 100):.2f}%")
    print("=" * 70 + "\n")
    
    # 5. Vẽ biểu đồ
    if MATPLOTLIB_AVAILABLE:
        print("Plotting convergence and tour...")
        ga_solver.plot_convergence()
        ga_solver.plot_tour()
    else:
        print("Matplotlib not available. Skipping plots.")


if __name__ == "__main__":
    demo_ga()
