"""
Particle Swarm Optimization (PSO) for Traveling Salesman Problem (TSP)

Thuật toán PSO cải tiến (weighted PSO) cho bài toán TSP rời rạc.
Hỗ trợ cả gbest PSO và lbest PSO.
Sử dụng biểu diễn hoán vị (permutation) để thể hiện hạt = đáp án đường đi Hamilton.

Tham khảo:
- Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization.
- Shi, Y., et al. (2007). Particle swarm optimization for traveling salesman problem.

Các phép toán định nghĩa cho PSO rời rạc:
- Phép trừ vị trí: X_j - X_i = V (chuỗi chuyển vị ngắn nhất)
- Phép cộng vận tốc: V_1 + V_2 = chuỗi chuyển vị nối tiếp
- Phép nhân vô hướng: V · r = lấy r phần của chuỗi chuyển vị
- Phép cộng vị trí và vận tốc: X + V = áp dụng chuỗi chuyển vị lên X

Topology:
- gbest PSO: Tất cả hạt bị ảnh hưởng bởi vị trí tốt nhất toàn cục
- lbest PSO: Mỗi hạt chỉ bị ảnh hưởng bởi vị trí tốt nhất trong lân cận
"""

import numpy as np
import random
import time
from typing import List, Tuple, Optional
try:
    from .tsp_problem import TSPProblem, create_tsp
except ImportError:
    from tsp_problem import TSPProblem, create_tsp
import matplotlib.pyplot as plt


class Particle:
    """
    Một hạt (particle) trong PSO cho TSP.
    
    Attributes:
        position: Vị trí hiện tại (hoán vị của các thành phố)
        velocity: Vận tốc (chuỗi các phép chuyển vị)
        pbest_position: Vị trí tốt nhất cá nhân (personal best)
        pbest_cost: Chi phí tốt nhất cá nhân
    """
    
    def __init__(self, position: List[int]):
        """
        Khởi tạo một hạt.
        
        Args:
            position: Vị trí ban đầu (hoán vị của các thành phố)
        """
        self.position = position.copy()
        self.velocity = []  # Chuỗi các phép chuyển vị (transpositions)
        self.pbest_position = position.copy() # Vị trí tốt nhất cá nhân
        self.pbest_cost = float('inf') # Chi phí tốt nhất cá nhân


class PSO:
    """
    Thuật toán Particle Swarm Optimization cho bài toán TSP.
    
    Hỗ trợ hai phiên bản:
    - weighted gbest PSO: Tất cả hạt bị ảnh hưởng bởi global best
    - weighted lbest PSO: Mỗi hạt bị ảnh hưởng bởi local best trong lân cận
    
    Attributes:
        tsp: Đối tượng bài toán TSP
        n_particles: Số lượng hạt trong bầy (swarm size)
        n_iterations: Số vòng lặp tối đa
        w: Trọng số quán tính (inertia weight)
        c1: Hệ số nhận thức (cognitive coefficient)
        c2: Hệ số xã hội (social coefficient)
        topology: Cấu trúc lân cận ('gbest' hoặc 'lbest')
        neighborhood_size: Kích thước lân cận cho lbest (chỉ dùng khi topology='lbest')
        particles: Danh sách các hạt
        gbest_position: Vị trí tốt nhất toàn cục (global best)
        gbest_cost: Chi phí tốt nhất toàn cục
    """
    
    def __init__(
        self,
        tsp_problem: TSPProblem,
        n_particles: int = 30,
        n_iterations: int = 100,
        w: float = 0.7,
        c1: float = 0.9,
        c2: float = 0.3,
        topology: str = 'gbest',
        neighborhood_size: int = 5,
        seed: Optional[int] = None
    ):
        """
        Khởi tạo PSO solver.
        
        Args:
            tsp_problem: Bài toán TSP cần giải
            n_particles: Số lượng hạt trong bầy (>= 2)
            n_iterations: Số vòng lặp (>= 1)
            w: Trọng số quán tính (0 < w < 1)
            c1: Hệ số nhận thức (0 < c1 < 1)
            c2: Hệ số xã hội (0 < c2 < 1)
            topology: Cấu trúc lân cận ('gbest' hoặc 'lbest')
            neighborhood_size: Kích thước lân cận (cho lbest, phải >= 2)
            seed: Random seed
        """
        # Kiểm tra tham số
        if n_particles < 2:
            raise ValueError(f"n_particles phải >= 2, nhận được: {n_particles}")
        if n_iterations < 1:
            raise ValueError(f"n_iterations phải >= 1, nhận được: {n_iterations}")
        if not 0 < w < 1:
            raise ValueError(f"w phải trong khoảng (0, 1), nhận được: {w}")
        if not 0 < c1 < 1 or not 0 < c2 < 1:
            raise ValueError(f"c1 và c2 phải trong khoảng (0, 1), nhận được: c1={c1}, c2={c2}")
        if topology not in ['gbest', 'lbest']:
            raise ValueError(f"topology phải là 'gbest' hoặc 'lbest', nhận được: {topology}")
        if topology == 'lbest':
            if neighborhood_size < 2:
                raise ValueError(f"neighborhood_size phải >= 2, nhận được: {neighborhood_size}")
            if neighborhood_size > n_particles:
                raise ValueError(f"neighborhood_size không được lớn hơn n_particles")
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.tsp = tsp_problem
        self.n_cities = tsp_problem.n_cities
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.topology = topology
        self.neighborhood_size = neighborhood_size
        
        # Khởi tạo bầy hạt
        self.particles = []
        self._initialize_swarm()
        
        # Global best
        self.gbest_position = None
        self.gbest_cost = float('inf')
        
        # Lịch sử chi phí
        self.cost_history = []
    
    def _initialize_swarm(self):
        """Khởi tạo bầy hạt với các vị trí ngẫu nhiên."""
        for _ in range(self.n_particles):
            position = self.tsp.generate_random_tour()
            particle = Particle(position)
            self.particles.append(particle)
    
    def _calculate_transposition_sequence(
        self, 
        source: List[int], 
        target: List[int]
    ) -> List[Tuple[int, int]]:
        """
        Tính chuỗi chuyển vị từ source đến target.
        
        Phép trừ vị trí: target - source = V
        Trả về chuỗi chuyển vị ngắn nhất để biến source thành target (hoặc SL/RE của target).
        
        Args:
            source: Hoán vị nguồn
            target: Hoán vị đích
        
        Returns:
            Danh sách các phép chuyển vị [(i1, j1), (i2, j2), ...]
            Mỗi phép chuyển vị (i, j) hoán đổi phần tử ở vị trí i và j
        """
        # Thử cả target gốc và các biến thể (SL, RE)
        candidates = [target]
        
        # Thêm các phép dịch vòng (SL)
        for k in range(1, len(target)):
            shifted = target[k:] + target[:k]
            candidates.append(shifted)
        
        # Thêm phép đảo ngược (RE)
        reversed_target = target[::-1]
        candidates.append(reversed_target)
        
        # Thêm RE + SL
        for k in range(1, len(reversed_target)):
            shifted_rev = reversed_target[k:] + reversed_target[:k]
            candidates.append(shifted_rev)
        
        # Tìm chuỗi chuyển vị ngắn nhất
        best_sequence = None
        min_length = float('inf')
        
        for candidate in candidates:
            sequence = self._find_transposition_sequence(source, candidate)
            if len(sequence) < min_length:
                min_length = len(sequence)
                best_sequence = sequence
        
        return best_sequence
    
    def _find_transposition_sequence(
        self, 
        source: List[int], 
        target: List[int]
    ) -> List[Tuple[int, int]]:
        """
        Tìm chuỗi chuyển vị để biến source thành target.
        
        Args:
            source: Hoán vị nguồn
            target: Hoán vị đích
        
        Returns:
            Danh sách các phép chuyển vị
        """
        current = source.copy()
        sequence = []
        
        for i in range(len(current)):
            if current[i] != target[i]:
                # Tìm vị trí của target[i] trong current
                j = current.index(target[i])
                # Hoán đổi
                current[i], current[j] = current[j], current[i]
                sequence.append((i, j))
        
        return sequence
    
    def _add_velocities(
        self, 
        v1: List[Tuple[int, int]], 
        v2: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """
        Phép cộng vận tốc: V1 + V2 = nối chuỗi chuyển vị.
        
        Args:
            v1: Chuỗi chuyển vị thứ nhất
            v2: Chuỗi chuyển vị thứ hai
        
        Returns:
            Chuỗi chuyển vị kết hợp
        """
        return v1 + v2
    
    def _multiply_velocity(
        self, 
        velocity: List[Tuple[int, int]], 
        scalar: float
    ) -> List[Tuple[int, int]]:
        """
        Phép nhân vô hướng: V · r = lấy r phần của chuỗi chuyển vị.
        
        Args:
            velocity: Chuỗi chuyển vị
            scalar: Hệ số nhân (0 <= r <= 1)
        
        Returns:
            Chuỗi chuyển vị đã nhân
        """
        if not velocity:
            return []
        
        # Giới hạn scalar trong [0, 1]
        scalar = max(0.0, min(1.0, scalar))
        
        # Tính số phép chuyển vị cần giữ lại
        length = int(np.floor(scalar * len(velocity)))
        
        return velocity[:length]
    
    def _apply_velocity(
        self, 
        position: List[int], 
        velocity: List[Tuple[int, int]]
    ) -> List[int]:
        """
        Phép cộng vị trí và vận tốc: X + V = áp dụng chuỗi chuyển vị lên X.
        
        Args:
            position: Vị trí hiện tại (hoán vị)
            velocity: Vận tốc (chuỗi chuyển vị)
        
        Returns:
            Vị trí mới sau khi áp dụng vận tốc
        """
        new_position = position.copy()
        
        for (i, j) in velocity:
            # Đảm bảo chỉ số hợp lệ
            if 0 <= i < len(new_position) and 0 <= j < len(new_position):
                new_position[i], new_position[j] = new_position[j], new_position[i]
        
        return new_position
    
    def _get_neighborhood(self, particle_index: int) -> List[int]:
        """
        Lấy chỉ số các hạt trong lân cận của hạt tại particle_index.
        
        Định nghĩa lân cận theo ring topology:
        N_i = {y_{i-n}, y_{i-n+1}, ..., y_{i-1}, y_i, y_{i+1}, ..., y_{i+n}}
        với n = neighborhood_size (mở rộng về 2 đầu)
        
        Tổng số hạt trong lân cận: 2*n + 1 (bao gồm cả hạt hiện tại)
        
        Ví dụ:
        - neighborhood_size=5: lấy 5 hạt trái + hạt hiện tại + 5 hạt phải = 11 hạt
        - neighborhood_size=3: lấy 3 hạt trái + hạt hiện tại + 3 hạt phải = 7 hạt
        
        Args:
            particle_index: Chỉ số của hạt cần tính lân cận
        
        Returns:
            Danh sách chỉ số các hạt trong lân cận (bao gồm cả hạt hiện tại)
        """
        n = self.neighborhood_size
        neighborhood = []
        
        for offset in range(-n, n + 1):
            # Sử dụng modulo để tạo ring topology
            neighbor_index = (particle_index + offset) % self.n_particles
            neighborhood.append(neighbor_index)
        
        return neighborhood
    
    def _get_local_best(self, particle_index: int) -> Tuple[List[int], float]:
        """
        Tìm vị trí tốt nhất trong lân cận của một hạt (local best).
        
        Args:
            particle_index: Chỉ số của hạt
        
        Returns:
            Tuple (vị trí local best, chi phí local best)
        """
        neighborhood_indices = self._get_neighborhood(particle_index)
        
        best_position = None
        best_cost = float('inf')
        
        for idx in neighborhood_indices:
            if self.particles[idx].pbest_cost < best_cost:
                best_cost = self.particles[idx].pbest_cost
                best_position = self.particles[idx].pbest_position.copy()
        
        return best_position, best_cost
    
    def _update_particle(self, particle: Particle, particle_index: int):
        """
        Cập nhật vận tốc và vị trí của một hạt theo công thức PSO.
        
        Công thức vận tốc:
        - gbest PSO: v_i(t+1) = w·v_i(t) + c1·r1·(pbest_i - x_i) + c2·r2·(gbest - x_i)
        - lbest PSO: v_i(t+1) = w·v_i(t) + c1·r1·(pbest_i - x_i) + c2·r2·(lbest_i - x_i)
        
        Công thức vị trí:
        x_i(t+1) = x_i(t) + v_i(t+1)
        
        Args:
            particle: Hạt cần cập nhật
            particle_index: Chỉ số của hạt trong danh sách
        """
        # Tạo các hệ số ngẫu nhiên
        r1 = random.random()
        r2 = random.random()
        
        # Thành phần quán tính: w * v_i(t)
        inertia = self._multiply_velocity(particle.velocity, self.w)
        
        # Thành phần nhận thức: c1 * r1 * (pbest - x_i)
        cognitive_velocity = self._calculate_transposition_sequence(
            particle.position, 
            particle.pbest_position
        )
        cognitive = self._multiply_velocity(cognitive_velocity, self.c1 * r1)
        
        # Thành phần xã hội: c2 * r2 * (best - x_i)
        # Sử dụng gbest hoặc lbest tùy theo topology
        if self.topology == 'gbest':
            best_position = self.gbest_position
        else:  # lbest
            best_position, _ = self._get_local_best(particle_index)
        
        social_velocity = self._calculate_transposition_sequence(
            particle.position,
            best_position
        )
        social = self._multiply_velocity(social_velocity, self.c2 * r2)
        
        # Cập nhật vận tốc: v = w*v + c1*r1*(pbest-x) + c2*r2*(gbest-x)
        new_velocity = self._add_velocities(inertia, cognitive)
        new_velocity = self._add_velocities(new_velocity, social)
        particle.velocity = new_velocity
        
        # Cập nhật vị trí: x = x + v
        particle.position = self._apply_velocity(particle.position, particle.velocity)
    
    def solve(self) -> Tuple[List[int], float, List[float], float]:
        """
        Thực thi thuật toán PSO để tìm giải pháp cho TSP.
        
        Returns:
            Tuple chứa (lộ trình tốt nhất, chi phí tốt nhất, lịch sử chi phí, thời gian thực hiện)
        """
        topology_name = "gbest" if self.topology == 'gbest' else f"lbest (neighborhood={self.neighborhood_size})"
        print(f"Starting PSO for TSP (topology: {topology_name})...")
        start_time = time.time()
        
        # Khởi tạo pbest và gbest
        for particle in self.particles:
            cost = self.tsp.calculate_tour_cost(particle.position)
            particle.pbest_cost = cost
            
            if cost < self.gbest_cost:
                self.gbest_cost = cost
                self.gbest_position = particle.position.copy()
        
        # Vòng lặp chính
        for iteration in range(self.n_iterations):
            # Cập nhật từng hạt
            for i, particle in enumerate(self.particles):
                # Cập nhật vận tốc và vị trí (truyền index cho lbest)
                self._update_particle(particle, i)
                
                # Đánh giá vị trí mới
                cost = self.tsp.calculate_tour_cost(particle.position)
                
                # Cập nhật pbest
                if cost < particle.pbest_cost:
                    particle.pbest_cost = cost
                    particle.pbest_position = particle.position.copy()
                
                # Cập nhật gbest
                if cost < self.gbest_cost:
                    self.gbest_cost = cost
                    self.gbest_position = particle.position.copy()
            
            # Lưu lịch sử
            self.cost_history.append(self.gbest_cost)
            
            # In tiến trình
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.n_iterations}, Best cost: {self.gbest_cost:.2f}")
        
        print("Finished.")
        
        end_time = time.time()
        self.elapsed_time = end_time - start_time
        
        return self.gbest_position, self.gbest_cost, self.cost_history, self.elapsed_time
    
    def plot_convergence(self):
        """Vẽ biểu đồ hội tụ của thuật toán PSO với thông tin siêu tham số."""
        if not self.cost_history:
            print("No cost history available for plotting.")
            return
        
        plt.figure(figsize=(10, 7))
        plt.plot(self.cost_history, 'b-', linewidth=2, label='Best Cost')
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Best Cost', fontsize=12)
        
        # Tạo tiêu đề với thông tin siêu tham số
        topology_str = f"Topology: {self.topology}"
        if self.topology == 'lbest':
            topology_str += f" (k={self.neighborhood_size})"
        title = f'PSO Convergence - {topology_str}\n(Cities: {self.n_cities}, Particles: {self.n_particles}, Iterations: {self.n_iterations})'
        plt.title(title, fontsize=14, fontweight='bold')
        
        # Thêm text box với các siêu tham số
        param_text = f'Topology: {self.topology}'
        if self.topology == 'lbest':
            param_text += f'\nNeighborhood: {self.neighborhood_size}'
        param_text += f'\nw={self.w}, c1={self.c1}, c2={self.c2}'
        if hasattr(self, 'elapsed_time'):
            param_text += f'\nTime: {self.elapsed_time:.2f}s'
        param_text += f'\nBest Cost: {self.gbest_cost:.2f}'
        
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
            tour: Lộ trình cần vẽ. Nếu None, vẽ gbest_position.
            title: Tiêu đề của biểu đồ. Nếu None, tự động tạo tiêu đề.
        """
        if tour is None:
            tour = self.gbest_position
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
            title = f"Best TSP Tour Found by PSO\nCost: {tour_cost:.2f}"
        
        plt.figure(figsize=(10, 7))
        
        # Vẽ các thành phố
        plt.scatter(positions[1:, 0], positions[1:, 1], c='blue', s=100, zorder=3, label='Cities')
        for i in range(1, self.tsp.n_cities + 1):
            plt.text(positions[i, 0], positions[i, 1] + 2, str(i), fontsize=10, ha='center')
        
        # Vẽ lộ trình
        tour_positions = np.array([positions[city] for city in tour])
        tour_positions = np.vstack([tour_positions, tour_positions[0]])  # Nối điểm cuối về đầu
        
        plt.plot(tour_positions[:, 0], tour_positions[:, 1], 'r-', linewidth=2,
                zorder=2, label=f'PSO Tour (Cost: {tour_cost:.2f})')
        
        # Đánh dấu điểm bắt đầu
        start_pos = positions[tour[0]]
        plt.scatter(start_pos[0], start_pos[1], c='green', s=200, marker='*',
                   zorder=4, label=f'Start (City {tour[0]})')
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel("X-coordinate", fontsize=12)
        plt.ylabel("Y-coordinate", fontsize=12)
        
        # Thêm text box với các siêu tham số
        param_text = f'Hyperparameters:\n'
        param_text += f'Topology: {self.topology}'
        if self.topology == 'lbest':
            param_text += f' (k={self.neighborhood_size})'
        param_text += f'\nParticles: {self.n_particles}, Iterations: {self.n_iterations}\n'
        param_text += f'w={self.w}, c1={self.c1}, c2={self.c2}'
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


def demo_pso():
    """Demo thuật toán PSO cho TSP với cả gbest và lbest."""
    print("=" * 70)
    print("PSO FOR TSP - DEMO")
    print("=" * 70)
    
    # 1. Tạo bài toán TSP
    N_CITIES = 30
    tsp = create_tsp(n_cities=N_CITIES, symmetric=True, seed=42)
    print(f"\nCreated TSP with {N_CITIES} cities.")
    
    # 2. Định nghĩa các siêu tham số cho PSO
    N_PARTICLES = 150      # Số hạt trong bầy (>= 2)
    N_ITERATIONS = 100    # Số vòng lặp (>= 1)
    W = 0.3              # Trọng số quán tính (0 < w < 1)
    C1 = 0.5             # Hệ số nhận thức (0 < c1 < 1)
    C2 = 0.9             # Hệ số xã hội (0 < c2 < 1)
    NEIGHBORHOOD_SIZE = 15 # Kích thước lân cận cho lbest (>= 2)

    # ===== DEMO 1: GBEST PSO =====
    print("\n" + "="*70)
    print("DEMO 1: GLOBAL BEST PSO (gbest)")
    print("="*70)
    
    pso_gbest = PSO(
        tsp_problem=tsp,
        n_particles=N_PARTICLES,
        n_iterations=N_ITERATIONS,
        w=W,
        c1=C1,
        c2=C2,
        topology='gbest',
        seed=42
    )
    
    best_tour_gbest, best_cost_gbest, cost_history_gbest, elapsed_time_gbest = pso_gbest.solve()
    
    print("\n" + "="*50)
    print("RESULTS OF GBEST PSO")
    print(f"Best tour: {best_tour_gbest}")
    print(f"Best cost: {best_cost_gbest:.2f}")
    print(f"Elapsed time: {elapsed_time_gbest:.2f} seconds")
    print("="*50 + "\n")

    # ===== DEMO 2: LBEST PSO =====
    print("\n" + "="*70)
    print(f"DEMO 2: LOCAL BEST PSO (lbest) - Neighborhood size: {NEIGHBORHOOD_SIZE}")
    print("="*70)
    
    pso_lbest = PSO(
        tsp_problem=tsp,
        n_particles=N_PARTICLES,
        n_iterations=N_ITERATIONS,
        w=W,
        c1=C1,
        c2=C2,
        topology='lbest',
        neighborhood_size=NEIGHBORHOOD_SIZE,
        seed=42
    )
    
    best_tour_lbest, best_cost_lbest, cost_history_lbest, elapsed_time_lbest = pso_lbest.solve()
    
    print("\n" + "="*50)
    print("RESULTS OF LBEST PSO")
    print(f"Best tour: {best_tour_lbest}")
    print(f"Best cost: {best_cost_lbest:.2f}")
    print(f"Elapsed time: {elapsed_time_lbest:.2f} seconds")
    print("="*50 + "\n")
    
    # ===== SO SÁNH KẾT QUẢ =====
    print("\n" + "="*70)
    print("COMPARISON: GBEST vs LBEST")
    print("="*70)
    
    random_tour = tsp.generate_random_tour()
    random_cost = tsp.calculate_tour_cost(random_tour)
    
    print(f"\nRandom tour cost: {random_cost:.2f}")
    print(f"\nGbest PSO:")
    print(f"  - Best cost: {best_cost_gbest:.2f}")
    print(f"  - Improvement: {((random_cost - best_cost_gbest) / random_cost * 100):.2f}%")
    print(f"  - Time: {elapsed_time_gbest:.2f}s")
    
    print(f"\nLbest PSO (k={NEIGHBORHOOD_SIZE}):")
    print(f"  - Best cost: {best_cost_lbest:.2f}")
    print(f"  - Improvement: {((random_cost - best_cost_lbest) / random_cost * 100):.2f}%")
    print(f"  - Time: {elapsed_time_lbest:.2f}s")
    
    winner = "Gbest" if best_cost_gbest < best_cost_lbest else "Lbest"
    diff = abs(best_cost_gbest - best_cost_lbest)
    print(f"\nWinner: {winner} PSO (difference: {diff:.2f})")
    print("="*70 + "\n")
    
    # ===== VẼ BIỂU ĐỒ SO SÁNH HỘI TỤ =====
    plt.figure(figsize=(12, 7))
    plt.plot(cost_history_gbest, 'b-', linewidth=2, label='Gbest PSO')
    plt.plot(cost_history_lbest, 'r-', linewidth=2, label=f'Lbest PSO (k={NEIGHBORHOOD_SIZE})')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Best Cost', fontsize=12)
    plt.title(f'PSO Convergence Comparison\n(Cities: {N_CITIES}, Particles: {N_PARTICLES}, Iterations: {N_ITERATIONS})', 
              fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # ===== VẼ BIỂU ĐỒ RIÊNG LẺ =====
    print("Plotting individual convergence charts and tours...")
    # pso_gbest.plot_convergence()
    pso_gbest.plot_tour()
    
    # pso_lbest.plot_convergence()
    pso_lbest.plot_tour()


if __name__ == "__main__":
    demo_pso()
