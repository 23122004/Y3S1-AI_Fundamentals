"""
Traveling Salesman Problem (TSP) Environment
Bài toán người bán hàng du lịch

Định nghĩa toán học:
- Cho đồ thị G = (V, A), với V = {1, 2, ..., n} là tập hợp gồm n đỉnh
- A là tập hợp các cung hoặc cạnh
- Ma trận khoảng cách (hoặc chi phí) C = (c_ij) ứng với A
- c_ij là chi phí để đi trực tiếp từ đỉnh i đến đỉnh j

Biến quyết định:
    x_ij = 1 nếu lộ trình có đi trực tiếp từ i đến j
    x_ij = 0 ngược lại

Hàm mục tiêu:
    min Z = Σ_{i∈V} Σ_{j∈V, j≠i} c_ij * x_ij

Ràng buộc (đảm bảo chu trình Hamilton):
    1. Σ_{j∈V, j≠i} x_ij = 1, ∀i∈V  (mỗi đỉnh có đúng 1 cung đi ra)
    2. Σ_{i∈V, i≠j} x_ij = 1, ∀j∈V  (mỗi đỉnh có đúng 1 cung đi vào)
    3. x_ij ∈ {0,1}, ∀i,j∈V, i≠j
    4. u_i - u_j + (n-1)*x_ij ≤ n-2, ∀i,j∈V\\{1}, i≠j  (loại bỏ chu trình con)
       với u_1 = 0; 1 ≤ u_i ≤ n-1, ∀i∈V\\{1}

Môi trường bài toán TSP giúp đánh giá các thuật giải:
- Genetic Algorithm (GA)
- Particle Swarm Optimization (PSO)
- Ant Colony Optimization (ACO)
"""

import numpy as np
from typing import List, Optional
import random
from pathlib import Path

class TSPProblem:
    """
    Môi trường bài toán TSP
    
    Attributes:
        n_cities: Số lượng đỉnh n
        cost_matrix: Ma trận khoảng cách/chi phí C = (c_ij), kích thước (n+1)×(n+1)
                     với chỉ số từ 0 đến n (sử dụng chỉ số 1..n, bỏ qua chỉ số 0)
        city_positions: Tọa độ thành phố (nếu có), kích thước (n+1)×2
        V: Tập đỉnh {1, 2, ..., n}
        
    Note:
        Ma trận cost_matrix có kích thước (n+1)×(n+1)
        để có thể sử dụng trực tiếp chỉ số 1..n mà không cần chuyển đổi.
        Hàng 0 và cột 0 của ma trận sẽ không được sử dụng.
    """
    
    def __init__(
        self, 
        n_cities: int, 
        cost_matrix: Optional[np.ndarray] = None, 
        city_positions: Optional[np.ndarray] = None, 
        seed: Optional[int] = None
    ):
        """
        Khởi tạo bài toán TSP
        
        Args:
            n_cities: Số lượng đỉnh n (≥ 3)
            cost_matrix: Ma trận chi phí (n+1)×(n+1), chỉ số 1..n được sử dụng. 
                        Nếu None, tạo từ positions
            city_positions: Tọa độ thành phố (n+1)×2, chỉ số 1..n được sử dụng. 
                        Nếu None, tạo ngẫu nhiên
            seed: Random seed để tái tạo kết quả
        
        Note:
            Tập đỉnh V = {1, 2, ..., n}. Hàng 0 và cột 0 của các ma trận không được sử dụng.
            Input chỉ cần cung cấp ma trận chi phí hoặc tọa độ thành phố, 
            nếu cả hai đều None thì tạo ngẫu nhiên tọa độ.
        """
        if n_cities < 3:
            raise ValueError("Number of cities must be ≥ 3")
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        self.n_cities = n_cities
        self.V = list(range(1, n_cities + 1))  # V = {1, 2, ..., n}
        
        # Xác định ma trận chi phí và tọa độ (ưu tiên cost_matrix)
        if cost_matrix is not None:
            self.cost_matrix = np.array(cost_matrix)
            self.city_positions = None if city_positions is None else np.array(city_positions)
        elif city_positions is not None:
            self.city_positions = np.array(city_positions)
            self.cost_matrix = self._compute_distance_matrix()
        else:
            # Tạo random positions trong không gian [0, 100]×[0, 100]
            # Kích thước (n+1)×2, sử dụng chỉ số 1..n
            self.city_positions = np.zeros((n_cities + 1, 2))
            self.city_positions[1:] = np.random.rand(n_cities, 2) * 100
            self.cost_matrix = self._compute_distance_matrix()
        
        self._validate()
    
    def _compute_distance_matrix(self) -> np.ndarray:
        """
        Tính ma trận khoảng cách Euclidean từ tọa độ
        
        Returns:
            Ma trận kích thước (n+1)×(n+1), với c_ij là khoảng cách giữa đỉnh i và j
            Chỉ số 1..n được sử dụng, hàng 0 và cột 0 được đặt bằng 0
        """
        n = self.n_cities
        dist_matrix = np.zeros((n + 1, n + 1))
        
        for i in range(1, n + 1):
            for j in range(i + 1, n + 1):
                dist = np.linalg.norm(self.city_positions[i] - self.city_positions[j])
                dist_matrix[i, j] = dist_matrix[j, i] = dist
        
        return dist_matrix
    
    def _validate(self):
        """
        Kiểm tra tính hợp lệ của dữ liệu đầu vào
        
        Ràng buộc:
        - Ma trận chi phí phải có kích thước (n+1)×(n+1)
        - c_ii = 0, ∀i∈V (đường chéo chính bằng 0)
        - Tọa độ (nếu có) phải có kích thước (n+1)×2
        """
        expected_size = self.n_cities + 1
        
        if self.cost_matrix.shape != (expected_size, expected_size):
            raise ValueError(f"Cost matrix must be {expected_size}×{expected_size}")

        if not np.allclose(np.diag(self.cost_matrix), 0):
            raise ValueError("Main diagonal of cost matrix must be 0 (c_ii = 0, ∀i)")

        if self.city_positions is not None and self.city_positions.shape != (expected_size, 2):
            raise ValueError(f"City positions must have size {expected_size}×2")
    
    # ============================================================
    # CORE TSP METHODS
    # ============================================================
    
    def calculate_tour_cost(self, tour: List[int]) -> float:
        """
        Tính tổng chi phí của chu trình
        
        Hàm mục tiêu: Z = Σ_{i∈V} Σ_{j∈V, j≠i} c_ij * x_ij
        
        Với chu trình đã cho, x_ij = 1 nếu đi trực tiếp từ i đến j, ngược lại x_ij = 0
        
        Args:
            tour: Chu trình [city_1, city_2, ..., city_n], mỗi city_i ∈ V = {1,2,...,n}
        
        Returns:
            Tổng chi phí của chu trình (inf nếu tour không hợp lệ)
        """
        if not self.is_valid_tour(tour):
            return float('inf')
        
        cost = sum(
            self.cost_matrix[tour[i], tour[(i + 1) % len(tour)]]
            for i in range(len(tour))
        )
        return cost
    
    def is_valid_tour(self, tour: List[int]) -> bool:
        """
        Kiểm tra chu trình Hamilton hợp lệ
        
        Ràng buộc để đảm bảo chu trình Hamilton:
        1. Độ dài = n (số đỉnh)
        2. Mỗi đỉnh xuất hiện đúng 1 lần (Σ x_ij = 1, Σ x_ji = 1)
        3. Tất cả đỉnh thuộc V = {1, 2, ..., n}
        4. Không có chu trình con (đảm bảo bởi ràng buộc u_i)
        5. Không có cạnh nào trong tour có chi phí = inf (không tồn tại đường đi)
        
        Args:
            tour: Danh sách các đỉnh trong chu trình
            
        Returns:
            True nếu chu trình hợp lệ, False ngược lại
        """
        # Kiểm tra cấu trúc cơ bản của tour
        if not (len(tour) == self.n_cities and
                len(set(tour)) == self.n_cities and
                set(tour) == set(self.V)):
            return False
        
        # Kiểm tra không có cạnh nào có chi phí inf
        for i in range(len(tour)):
            from_city = tour[i]
            to_city = tour[(i + 1) % len(tour)]
            if np.isinf(self.cost_matrix[from_city, to_city]):
                return False
        
        return True
    
    # ============================================================
    # SOLUTION GENERATION & EVALUATION
    # ============================================================
    
    def generate_random_tour(self) -> List[int]:
        """
        Tạo chu trình ngẫu nhiên
        
        Returns:
            Danh sách các đỉnh [i_1, i_2, ..., i_n] với i_k ∈ V = {1,2,...,n}
        """
        tour = self.V.copy()
        random.shuffle(tour)
        return tour
    
    def evaluate_solution(self, tour: List[int]) -> bool:
        """
        Đánh giá giải pháp chu trình: xác định tính hợp lệ, 
        biểu diễn đường đi và tính tổng chi phí
        
        Args:
            tour: Chu trình [city_1, city_2, ..., city_n]
        
        Returns:
            None
        """
        print("=" * 70)
        print("TSP SOLUTION EVALUATION")
        
        # Kiểm tra tính hợp lệ
        is_valid = self.is_valid_tour(tour)
        print(f"\n- Route: {tour}")
        print(f"- Valid (Hamilton route): {'Yes' if is_valid else 'No'}")

        # Tính chi phí
        cost = self.calculate_tour_cost(tour)
        
        # Hiển thị chi tiết đường đi
        print("\nRoute details:")
        for i in range(len(tour)):
            from_city = tour[i]
            to_city = tour[(i + 1) % len(tour)]
            edge_cost = self.cost_matrix[from_city, to_city]
            print(f"   Step {i+1}: {from_city} → {to_city}, cost c_{{{from_city}{to_city}}} = {edge_cost:.2f}")
        
        # Tổng chi phí
        print(f"\nTotal cost: Z = Σ c_ij * x_ij = {cost:.2f}")
        print("=" * 70)
        return is_valid

    # ============================================================
    # ACCESSOR METHODS
    # ============================================================
    
    def get_cost_matrix(self) -> np.ndarray:
        """Lấy bản sao của ma trận chi phí"""
        return self.cost_matrix.copy()
    
    def get_city_positions(self) -> Optional[np.ndarray]:
        """Lấy bản sao của tọa độ thành phố (nếu có)"""
        if self.city_positions is not None:
            return self.city_positions.copy()
        return None
    
    def __str__(self) -> str:
        return f"TSP({self.n_cities} cities)"
    
    def get_info(self) -> None:
        """In thông tin cơ bản về bài toán TSP"""
        print(f"TSP Problem with {self.n_cities} cities.")
        print("Cost Matrix:")
        print(self.cost_matrix[1:, 1:])  # Bỏ qua hàng và cột 0
        if self.city_positions is not None:
            print("City Positions:")
            print(self.city_positions[1:, :])  # Bỏ qua hàng 0

# ============================================================
# FACTORY FUNCTIONS
# ============================================================

def create_tsp(
    n_cities: Optional[int] = None,
    symmetric: bool = True,
    cost_matrix_file: Optional[str] = None,
    seed: Optional[int] = None
) -> TSPProblem:
    """
    Factory function để tạo bài toán TSP với 3 trường hợp
    
    Args:
        n_cities: Số lượng thành phố n (≥ 3). Bắt buộc nếu không dùng cost_matrix_file
        symmetric: True = TSP đối xứng (c_ij = c_ji), False = TSP bất đối xứng (c_ij ≠ c_ji)
                   Chỉ áp dụng khi không dùng cost_matrix_file
        cost_matrix_file: Đường dẫn file txt chứa ma trận chi phí.
                          Nếu được cung cấp, sẽ đọc ma trận từ file và bỏ qua symmetric
        seed: Random seed
    
    Returns:
        TSPProblem instance với V = {1, 2, ..., n}
    
    File format (cost_matrix_file):
        - Dòng đầu: số thành phố n
        - n dòng tiếp theo: ma trận chi phí n×n (chỉ số từ 1..n)
        - Ví dụ cho n=3:
          3
          0 10 15
          10 0 20
          15 20 0
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    # Trường hợp 1: Đọc ma trận chi phí từ file txt
    if cost_matrix_file is not None:
        try:
            with open(cost_matrix_file, 'r') as f:
                lines = f.readlines()
            
            # Đọc số thành phố từ dòng đầu
            n = int(lines[0].strip())
            
            if n < 3:
                raise ValueError("Number of cities must be ≥ 3")
            
            # Đọc ma trận chi phí n×n
            cost_matrix_raw = []
            for i in range(1, n + 1):
                row = [float(x) for x in lines[i].strip().split()]
                if len(row) != n:
                    raise ValueError(f"Row {i} must have {n} values")
                cost_matrix_raw.append(row)
            
            # Tạo ma trận (n+1)×(n+1), sử dụng chỉ số 1..n
            cost_matrix = np.zeros((n + 1, n + 1))
            cost_matrix[1:, 1:] = np.array(cost_matrix_raw)
            
            return TSPProblem(n, cost_matrix=cost_matrix, seed=seed)
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Cannot find file: {cost_matrix_file}")
        except Exception as e:
            raise ValueError(f"Error reading file {cost_matrix_file}: {str(e)}")

    # Trường hợp 2 & 3: Tạo ma trận ngẫu nhiên (symmetric hoặc asymmetric)
    if n_cities is None:
        raise ValueError("n_cities must be provided when not using cost_matrix_file")

    if n_cities < 3:
        raise ValueError("Number of cities must be ≥ 3")

    if symmetric:
        # Trường hợp 2: TSP đối xứng c_ij = c_ji (từ tọa độ Euclidean)
        # Tạo ma trận (n+1)×2, sử dụng chỉ số 1..n
        positions = np.zeros((n_cities + 1, 2))
        positions[1:] = np.random.rand(n_cities, 2) * 100
        return TSPProblem(n_cities, city_positions=positions, seed=seed)
    else:
        # Trường hợp 3: TSP bất đối xứng c_ij ≠ c_ji
        # Tạo ma trận (n+1)×(n+1), sử dụng chỉ số 1..n
        cost_matrix = np.zeros((n_cities + 1, n_cities + 1))
        cost_matrix[1:, 1:] = np.random.rand(n_cities, n_cities) * 100
        np.fill_diagonal(cost_matrix, 0)
        return TSPProblem(n_cities, cost_matrix=cost_matrix, seed=seed) 

# ============================================================
# DEMO
# ============================================================

def demo_tsp():
    print("=" * 70)
    print("TSP PROBLEM ENVIRONMENT - DEMO")
    print("=" * 70)

    # Demo 1: TSP symmetric with 5 cities
    print("\nDemo 1: TSP symmetric (5 cities)")
    tsp = create_tsp(n_cities=5, symmetric=True, seed=42)
    tsp.get_info()

    tour_valid = tsp.generate_random_tour()
    tsp.evaluate_solution(tour_valid)

    # Demo 2: TSP asymmetric (4 cities)
    print("\nDemo 2: TSP asymmetric (4 cities)")
    tsp_asym = create_tsp(n_cities=4, symmetric=False, seed=42)
    tsp_asym.get_info()
    tour_asym = [1, 2, 3, 4]
    tsp_asym.evaluate_solution(tour_asym)

    # Demo 3: Invalid tour
    print("\nDemo 3: Invalid tour")
    tour_invalid = [1, 2, 3]  # Missing vertex
    tsp.evaluate_solution(tour_invalid)

    # Demo 4: TSP from cost matrix file (if file exists)
    print("\nDemo 4: TSP from cost matrix file")
    try:
        current_file = Path(__file__)
        sample_file = current_file.parent / "tsp_cost_matrix.txt"

        with open(sample_file, 'w') as f:
            f.write("4\n")
            f.write("0 inf 15 20\n")
            f.write("10 0 35 25\n")
            f.write("15 35 0 30\n")
            f.write("20 25 30 0\n")
        
        tsp_file = create_tsp(cost_matrix_file=sample_file, seed=42)
        tsp_file.get_info()
        tour_file = [1, 2, 3, 4]
        tsp_file.evaluate_solution(tour_file)
        
        print(f"\n(File demo created at: {sample_file})")
    except Exception as e:
        print(f"Could not demo file reading: {e}")

if __name__ == "__main__":
    demo_tsp()
