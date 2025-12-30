from .base import Optimizer
import numpy as np

class HillClimbing(Optimizer):
    def __init__(self, benchmark_func, max_iter, step_size=0.1, random_state=None):
        """
        Args:
            benchmark_func (BenchmarkFunction): Hàm mục tiêu cần tối ưu hóa.
            max_iter (int): Tổng số vòng lặp thực hiện.
            step_size (float): Kích thước bước nhảy tối đa khi tạo nghiệm láng giềng.
            random_state (int hoặc np.random.RandomState, tùy chọn): Seed cho bộ sinh số ngẫu nhiên.
        """
        super().__init__(benchmark_func, max_iter, random_state)
        self.step_size = step_size
        self._population_history = []

    def solve(self):
        """
        Thực thi thuật toán Hill Climbing.

        Returns:
            Giá trị nghiệm tìm được và giá trị hàm mục tiêu tương ứng.
        """
        # Tạo nghiệm ban đầu ngẫu nhiên
        current_solution = self._random.uniform(self.lower_bound, self.upper_bound)
        current_fitness = self.benchmark_func.evaluate(current_solution)

        self._best_solution = current_solution
        self._best_fitness = current_fitness

        for _ in range(self.max_iter):
            # Tạo nghiệm láng giềng ngẫu nhiên bằng cách di chuyển một bước nhỏ
            neighbor = current_solution + self._random.uniform(-self.step_size, self.step_size, self.dim)
            neighbor = np.clip(neighbor, self.lower_bound, self.upper_bound)
            neighbor_fitness = self.benchmark_func.evaluate(neighbor)

            # Kiểm tra nghiệm mới có tốt hơn không
            if neighbor_fitness < current_fitness:
                current_solution = neighbor
                current_fitness = neighbor_fitness

            if current_fitness < self._best_fitness:
                self._best_solution = current_solution
                self._best_fitness = current_fitness

            # Lưu lịch sử tối ưu hóa (quá trình hội tụ)
            self._convergence_history.append(self._best_fitness)
            self._solution_history.append(self._best_solution)
            self._population_history.append(np.array([self._best_solution]))

        return self._best_solution, self._best_fitness
