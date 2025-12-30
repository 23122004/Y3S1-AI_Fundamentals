import numpy as np
import random, time
from typing import List, Tuple
try:
    from .tsp_problem import TSPProblem, create_tsp
except ImportError:
    from tsp_problem import TSPProblem, create_tsp
import matplotlib.pyplot as plt

class ACO:
    """
    Thuật toán Ant Colony Optimization để giải bài toán TSP.

    Sử dụng phiên bản Ant System (AS) gốc.

    Attributes:
        tsp (TSPProblem): Đối tượng bài toán TSP.
        n_ants (int): Số lượng kiến trong đàn (m).
        n_iterations (int): Số vòng lặp (thế hệ) để chạy thuật toán.
        alpha (float): Trọng số của pheromone.
        beta (float): Trọng số của thông tin heuristic.
        rho (float): Tốc độ bay hơi của pheromone (evaporation rate).
        Q (float): Hằng số pheromone, dùng để tính lượng pheromone kiến thả ra.
        
        pheromone (np.ndarray): Ma trận pheromone (τ).
        heuristic (np.ndarray): Ma trận thông tin heuristic (η).
        best_tour (List[int]): Lộ trình tốt nhất tìm được.
        best_cost (float): Chi phí của lộ trình tốt nhất.
    """

    def __init__(
        self,
        tsp_problem: TSPProblem,
        n_ants: int = 10,
        n_iterations: int = 100,
        alpha: float = 1.0,
        beta: float = 2.0,
        rho: float = 0.5,
        Q: float = 100.0,
    ):
        """Khởi tạo bộ giải ACO."""
        # Kiểm tra tham số đầu vào
        if not 0 <= rho <= 1:
            raise ValueError(f"rho phải trong khoảng [0, 1], nhận được: {rho}")
        if alpha < 0 or beta < 0:
            raise ValueError(f"alpha và beta phải >= 0, nhận được: alpha={alpha}, beta={beta}")
        if n_ants < 1 or n_iterations < 1:
            raise ValueError(f"n_ants và n_iterations phải >= 1, nhận được: n_ants={n_ants}, n_iterations={n_iterations}")
        if Q <= 0:
            raise ValueError(f"Q phải > 0, nhận được: {Q}")
        
        self.tsp = tsp_problem
        self.n_cities = tsp_problem.n_cities
        self.cost_matrix = tsp_problem.get_cost_matrix()

        # Phát hiện TSP đối xứng hay bất đối xứng
        self.symmetric = np.allclose(self.cost_matrix, self.cost_matrix.T, equal_nan=True)

        # Hyperparameters
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q

        # Khởi tạo ma trận Pheromone
        # Giá trị ban đầu nhỏ, dương để tất cả các cạnh có cơ hội được chọn
        initial_pheromone = 1.0 / (self.n_cities * np.mean(self.cost_matrix[self.cost_matrix > 0]))
        self.pheromone = np.full((self.n_cities + 1, self.n_cities + 1), initial_pheromone)

        # Khởi tạo ma trận Heuristic (η = 1/d_ij)
        # Thêm một số epsilon nhỏ để tránh chia cho 0
        self.heuristic = 1.0 / (self.cost_matrix + 1e-10)
        np.fill_diagonal(self.heuristic, 0)

        # Theo dõi giải pháp tốt nhất
        self.best_tour = None
        self.best_cost = float('inf')
        self.cost_history = []  # Lưu lịch sử chi phí tốt nhất mỗi iteration

    def solve(self) -> Tuple[List[int], float, List[float]]:
        """
        Thực thi thuật toán ACO để tìm giải pháp cho TSP.
        
        Returns:
            Tuple chứa (lộ trình tốt nhất, chi phí tốt nhất, lịch sử chi phí, thời gian thực hiện).
        """
        print("Starting ACO for TSP...")
        start_time = time.time()

        for iteration in range(self.n_iterations):
            all_tours = []
            
            # 1. Xây dựng giải pháp cho mỗi con kiến
            for _ in range(self.n_ants):
                tour = self._construct_tour_for_ant()
                cost = self.tsp.calculate_tour_cost(tour)
                all_tours.append((tour, cost))

                # Cập nhật giải pháp tốt nhất toàn cục
                if cost < self.best_cost:
                    self.best_cost = cost
                    self.best_tour = tour
            
            # Lưu lịch sử chi phí tốt nhất
            self.cost_history.append(self.best_cost)
            
            # 2. Cập nhật Pheromone
            self._update_pheromones(all_tours)

            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.n_iterations}, Best cost: {self.best_cost:.2f}")

        print("Finished.")

        end_time = time.time()
        self.elapsed_time = end_time - start_time

        return self.best_tour, self.best_cost, self.cost_history, self.elapsed_time

    def _update_pheromones(self, all_tours: List[Tuple[List[int], float]]):
        """
        Cập nhật ma trận pheromone sau một vòng lặp.
        
        Bao gồm 2 bước:
        1. Bay hơi (Evaporation): Giảm pheromone trên tất cả các cạnh.
        2. Tích tụ (Deposition): Thêm pheromone vào các cạnh được sử dụng bởi kiến.
        
        Note:
            - Với TSP đối xứng: cập nhật pheromone cho cả hai chiều (i→j và j→i)
            - Với TSP bất đối xứng: chỉ cập nhật pheromone cho chiều đi thực tế (i→j)
        """
        # 1. Bay hơi
        self.pheromone *= (1 - self.rho)

        # 2. Tích tụ
        for tour, cost in all_tours:
            # Bỏ qua tour không hợp lệ (chi phí inf hoặc không hữu hạn)
            if not np.isfinite(cost) or cost <= 0:
                continue
                
            pheromone_deposit = self.Q / cost
            for i in range(self.n_cities):
                from_city = tour[i]
                to_city = tour[(i + 1) % self.n_cities]
                
                # Cập nhật pheromone cho cạnh i→j
                self.pheromone[from_city, to_city] += pheromone_deposit
                
                # Chỉ cập nhật cho cả hai chiều nếu là TSP đối xứng
                if self.symmetric:
                    self.pheromone[to_city, from_city] += pheromone_deposit

    def _construct_tour_for_ant(self) -> List[int]:
        """Xây dựng một lộ trình hoàn chỉnh cho một con kiến."""
        # Bắt đầu từ một thành phố ngẫu nhiên
        start_city = random.choice(self.tsp.V)
        tour = [start_city]
        
        unvisited_cities = set(self.tsp.V)
        unvisited_cities.remove(start_city)

        while unvisited_cities:
            current_city = tour[-1]
            next_city = self._select_next_city(current_city, unvisited_cities)
            tour.append(next_city)
            unvisited_cities.remove(next_city)
            
        return tour

    def _select_next_city(self, current_city: int, unvisited_cities: set) -> int:
        """
        Chọn thành phố tiếp theo dựa trên xác suất pheromone và heuristic.
        
        p_ij ∝ (τ_ij^α) * (η_ij^β)
        """
        # Chuyển set thành list một lần để tối ưu hiệu suất
        cities_list = list(unvisited_cities)
        probabilities = []
        
        # Tính toán "sức hấp dẫn" của mỗi cạnh
        for city in cities_list:
            tau = self.pheromone[current_city, city] ** self.alpha
            eta = self.heuristic[current_city, city] ** self.beta
            prob = tau * eta
            probabilities.append(prob)

        total_prob = sum(probabilities)

        # Chuẩn hóa để tạo thành phân phối xác suất
        if total_prob == 0:
            # Nếu tất cả xác suất bằng 0, chọn ngẫu nhiên
            return random.choice(cities_list)
            
        probabilities = [p / total_prob for p in probabilities]

        # Chọn thành phố tiếp theo dựa trên bánh xe roulette
        selected_city = random.choices(cities_list, weights=probabilities, k=1)[0]
        return selected_city

    def plot_convergence(self):
        """Vẽ biểu đồ hội tụ của thuật toán ACO với thông tin siêu tham số."""
        if not self.cost_history:
            print("No cost history available for plotting.")
            return
        
        plt.figure(figsize=(10, 7))
        plt.plot(self.cost_history, 'b-', linewidth=2, label='Best Cost')
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Best Cost', fontsize=12)
        
        # Tạo tiêu đề với thông tin siêu tham số
        title = f'ACO Convergence (Cities: {self.n_cities}, Ants: {self.n_ants}, Iterations: {self.n_iterations})'
        plt.title(title, fontsize=14, fontweight='bold')
        
        # Thêm text box với các siêu tham số
        param_text = f'α={self.alpha}, β={self.beta}, ρ={self.rho}, Q={self.Q}'
        if hasattr(self, 'elapsed_time'):
            param_text += f'\nTime: {self.elapsed_time:.2f}s'
        param_text += f'\nBest Cost: {self.best_cost:.2f}'
        
        # Đặt text box ở góc trên phải
        plt.text(0.98, 0.98, param_text, 
                transform=plt.gca().transAxes,
                fontsize=10,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.legend(loc='upper right', bbox_to_anchor=(0.98, 0.85))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_tour(self, tour: List[int] = None, title: str = None):
        """
        Vẽ lộ trình TSP với thông tin siêu tham số.
        
        Args:
            tour: Lộ trình cần vẽ. Nếu None, vẽ best_tour.
            title: Tiêu đề của biểu đồ. Nếu None, tự động tạo tiêu đề.
        """
        if tour is None:
            tour = self.best_tour
            if tour is None:
                print("No tour available for plotting.")
                return
        
        positions = self.tsp.get_city_positions()
        if positions is None:
            print("No city positions available for plotting.")
            return

        # Tính chi phí của tour
        tour_cost = self.tsp.calculate_tour_cost(tour)
        
        # Tạo tiêu đề tự động nếu không được cung cấp
        if title is None:
            title = f"Best TSP Tour Found by ACO\nCost: {tour_cost:.2f}"

        plt.figure(figsize=(10, 7))

        # Vẽ các thành phố
        plt.scatter(positions[1:, 0], positions[1:, 1], c='blue', s=100, zorder=3, label='Cities')
        for i in range(1, self.tsp.n_cities + 1):
            plt.text(positions[i, 0], positions[i, 1] + 2, str(i), fontsize=10, ha='center')

        # Vẽ lộ trình
        tour_positions = np.array([positions[city] for city in tour])
        tour_positions = np.vstack([tour_positions, tour_positions[0]])  # Nối điểm cuối về đầu
        
        plt.plot(tour_positions[:, 0], tour_positions[:, 1], 'r-', linewidth=2, 
                zorder=2, label=f'ACO Tour (Cost: {tour_cost:.2f})')
        
        # Đánh dấu điểm bắt đầu
        start_pos = positions[tour[0]]
        plt.scatter(start_pos[0], start_pos[1], c='green', s=200, marker='*', 
                   zorder=4, label=f'Start (City {tour[0]})')
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel("X-coordinate", fontsize=12)
        plt.ylabel("Y-coordinate", fontsize=12)
        
        # Thêm text box với các siêu tham số
        param_text = f'Hyperparameters:\n'
        param_text += f'Ants: {self.n_ants}, Iterations: {self.n_iterations}\n'
        param_text += f'α={self.alpha}, β={self.beta}\n'
        param_text += f'ρ={self.rho}, Q={self.Q}'
        if hasattr(self, 'elapsed_time'):
            param_text += f'\nTime: {self.elapsed_time:.2f}s'
        
        # Đặt text box ở góc dưới trái
        plt.text(0.02, 0.02, param_text, 
                transform=plt.gca().transAxes,
                fontsize=9,
                verticalalignment='bottom',
                horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

def demo_aco():
    # 1. Tạo một bài toán TSP đối xứng với N_CITIES thành phố
    N_CITIES = 100
    tsp = create_tsp(n_cities=N_CITIES, symmetric=True, seed=42)
    print(f"Created TSP with {N_CITIES} cities.")

    # 2. Định nghĩa các siêu tham số cho ACO
    N_ANTS = 30         # Số kiến trong đàn (> 1)
    N_ITERATIONS = 50   # Số vòng lặp (> 1)
    ALPHA = 2.0         # Trọng số pheromone (>= 0)
    BETA = 5.0          # Trọng số heuristic (>= 0)
    RHO = 0.5           # Tốc độ bay hơi (0 < RHO < 1)
    Q = 100.0           # Hằng số pheromone (> 0)
    
    # 3. Khởi tạo và chạy thuật toán ACO
    aco_solver = ACO(
        tsp_problem=tsp,
        n_ants=N_ANTS,
        n_iterations=N_ITERATIONS,
        alpha=ALPHA,
        beta=BETA,
        rho=RHO,
        Q=Q,
    )

    best_tour, best_cost, cost_history, elapsed_time = aco_solver.solve()

    # 4. In kết quả
    print("\n" + "="*50)
    print("RESULTS OF ACO OPTIMIZATION")
    print(f"Best tour: {best_tour}")
    print(f"Best cost: {best_cost:.2f}")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print("="*50 + "\n")

    # 5. So sánh với một lộ trình ngẫu nhiên
    random_tour = tsp.generate_random_tour()
    random_cost = tsp.calculate_tour_cost(random_tour)
    print(f"Compare with random tour:")
    print(f"Random tour: {random_tour}")
    print(f"Random cost: {random_cost:.2f}")
    print(f"Improvement: {((random_cost - best_cost) / random_cost * 100):.2f}%\n")

    # 6. Vẽ biểu đồ hội tụ
    aco_solver.plot_convergence()
    
    # 7. Vẽ biểu đồ lộ trình tốt nhất
    aco_solver.plot_tour()

if __name__ == "__main__":
    demo_aco()