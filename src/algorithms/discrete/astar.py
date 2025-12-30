import numpy as np
import heapq
import time
from typing import List, Tuple
try:
    from .tsp_problem import TSPProblem
except ImportError:
    from tsp_problem import TSPProblem
import matplotlib.pyplot as plt

class AStar:
    """
    Thuật toán A* để giải bài toán Traveling Salesman Problem (TSP).

    Attributes:
        tsp (TSPProblem): Đối tượng bài toán TSP.
        n_cities (int): Số lượng thành phố.
        cost_matrix (np.ndarray): Ma trận chi phí.
        start_city (int): Thành phố bắt đầu, mặc định là 1.

        best_tour (List[int]): Lộ trình tốt nhất tìm được.
        best_cost (float): Chi phí của lộ trình tốt nhất.
        elapsed_time (float): Thời gian thực thi thuật toán.
        mst_cache (dict): Bộ nhớ đệm cho chi phí MST.
    """

    def __init__(self, tsp_problem: TSPProblem, start_city: int = 1):
        """
        Khởi tạo bộ giải A*.

        Args:
            tsp_problem (TSPProblem): Đối tượng bài toán TSP.
            start_city (int, tùy chọn): Thành phố bắt đầu cho lộ trình.
        """
        if start_city not in tsp_problem.V:
            raise ValueError(f"Start city {start_city} is not in the list of cities.")

        self.tsp = tsp_problem
        self.n_cities = tsp_problem.n_cities
        self.cost_matrix = tsp_problem.get_cost_matrix()
        self.start_city = start_city

        self.best_tour = None
        self.best_cost = float('inf')
        self.elapsed_time = 0.0
        self._mst_cache = {}

    def _heuristic_mst(self, current_city: int, unvisited: frozenset) -> float:
        """
        Heuristic (h): Ước tính chi phí còn lại bằng Cây bao trùm tối thiểu (MST).

        Sử dụng thuật toán Prim với min-heap để tính chi phí MST của các thành phố chưa được thăm
        cộng với thành phố hiện tại. Điều này đảm bảo heuristic là admissible.

        Args:
            current_city: Thành phố hiện tại.
            unvisited: Tập hợp các thành phố chưa được thăm.

        Returns:
            Chi phí ước tính để hoàn thành tour.
        """
        if not unvisited:
            # Chi phí quay về điểm bắt đầu
            return self.cost_matrix[current_city, self.start_city]

        cache_key = (current_city, unvisited)
        if cache_key in self._mst_cache:
            return self._mst_cache[cache_key]

        nodes = np.array(list(unvisited) + [current_city], dtype=np.int32)
        num_nodes = len(nodes)
        start_idx = num_nodes - 1  # Chỉ số của current_city trong nodes

        visited = np.zeros(num_nodes, dtype=bool)
        cost = np.full(num_nodes, np.inf, dtype=np.float64)
        cost[start_idx] = 0.0

        # Min-heap: (chi phí hiện tại, chỉ số đỉnh)
        heap = [(0.0, start_idx)]
        mst_cost = 0.0
        visited_count = 0

        while heap and visited_count < num_nodes:
            min_val, idx = heapq.heappop(heap)

            # Bỏ qua nếu đỉnh đã được thăm
            if visited[idx]:
                continue

            # Đánh dấu đỉnh là đã thăm
            visited[idx] = True
            mst_cost += min_val
            visited_count += 1

            # Lấy đỉnh tương ứng
            u_node = nodes[idx]

            # Cập nhật khoảng cách đến các đỉnh kề chưa thăm
            for v in range(num_nodes):
                if not visited[v]:
                    v_node = nodes[v]
                    edge_cost = self.cost_matrix[u_node, v_node]
                    if edge_cost < cost[v]:
                        cost[v] = edge_cost
                        heapq.heappush(heap, (edge_cost, v))

        self._mst_cache[cache_key] = mst_cost
        return mst_cost

    def solve(self) -> Tuple[List[int], float, float]:
        """
        Thực thi thuật toán A* để tìm giải pháp cho TSP.

        Returns:
            Tuple chứa (lộ trình tốt nhất, chi phí tốt nhất, thời gian thực hiện).
        """

        start_time = time.time()
        initial_path = (self.start_city,) # Tuple lưu lộ trình
        unvisited = frozenset(self.tsp.V) - {self.start_city}

        # Min-heap
        pq = [(0, 0, self.start_city, initial_path)] # (f, g, city, path)

        while pq:
            f_cost, g_cost, current_city, path = heapq.heappop(pq)

            # Nếu đã có giải pháp tốt hơn, bỏ qua nhánh này
            if g_cost >= self.best_cost:
                continue

            # Khi tất cả thành phố đã được thăm
            if len(path) == self.n_cities:
                final_cost = g_cost + self.cost_matrix[current_city, self.start_city]
                if final_cost < self.best_cost:
                    self.best_cost = final_cost
                    self.best_tour = list(path)
                continue

            unvisited = frozenset(self.tsp.V) - set(path)

            # Khám phá các thành phố chưa được thăm
            for next_city in unvisited:
                new_g_cost = g_cost + self.cost_matrix[current_city, next_city]

                # Ước tính chi phí còn lại
                remaining_unvisited = unvisited - {next_city}
                h_cost = self._heuristic_mst(next_city, remaining_unvisited)
                new_f_cost = new_g_cost + h_cost

                if new_f_cost < self.best_cost:
                    new_path = path + (next_city,)
                    heapq.heappush(pq, (new_f_cost, new_g_cost, next_city, new_path))

        end_time = time.time()
        self.elapsed_time = end_time - start_time

        return self.best_tour, self.best_cost, self.elapsed_time

    def plot_tour(self, tour: List[int] = None, title: str = None):
        """
        Vẽ lộ trình TSP.

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

        tour_cost = self.tsp.calculate_tour_cost(tour)

        if title is None:
            title = f"Best TSP Tour Found by A*\nCost: {tour_cost:.2f}"

        plt.figure(figsize=(10, 7))

        # Vẽ các thành phố
        plt.scatter(positions[1:, 0], positions[1:, 1], c='blue', s=100, zorder=3, label='Cities')
        for i in range(1, self.tsp.n_cities + 1):
            plt.text(positions[i, 0], positions[i, 1] + 2, str(i), fontsize=10, ha='center')

        # Vẽ lộ trình
        tour_positions = np.array([positions[city] for city in tour])
        tour_positions = np.vstack([tour_positions, tour_positions[0]])

        plt.plot(tour_positions[:, 0], tour_positions[:, 1], 'g-', linewidth=2,
                zorder=2, label=f'A* Tour (Cost: {tour_cost:.2f})')

        start_pos = positions[tour[0]]
        plt.scatter(start_pos[0], start_pos[1], c='red', s=200, marker='*',
                   zorder=4, label=f'Start (City {tour[0]})')

        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel("X-coordinate", fontsize=12)
        plt.ylabel("Y-coordinate", fontsize=12)

        param_text = f'Algorithm: A*\n'
        param_text += f'Cities: {self.n_cities}\n'
        param_text += f'Time: {self.elapsed_time:.2f}s'

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

if __name__ == "__main__":
    # Ví dụ sử dụng A* để giải TSP
    tsp = TSPProblem(15, seed=42)
    astar_solver = AStar(tsp_problem=tsp, start_city=1)
    best_tour, best_cost, elapsed_time = astar_solver.solve()

    print(f"Best Tour: {best_tour}")
    print(f"Best Cost: {best_cost}")
    print(f"Elapsed Time: {elapsed_time:.4f} seconds")

    astar_solver.plot_tour()
