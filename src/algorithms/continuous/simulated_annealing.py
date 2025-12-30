from .base import Optimizer
import numpy as np
import math

class SimulatedAnnealing(Optimizer):
    def __init__(self, benchmark_func, max_iter, step_size=0.1, initial_temp=100.0, cooling_rate=0.95, random_state=None):
        """
        Args:
            benchmark_func (BenchmarkFunction): Hàm mục tiêu cần tối ưu hóa.
            max_iter (int): Tổng số vòng lặp thực hiện.
            step_size (float): Kích thước bước nhảy tối đa khi tạo nghiệm láng giềng.
            initial_temp (float): Nhiệt độ ban đầu.
            cooling_rate (float): Tốc độ làm nguội nhiệt độ.
            random_state (int hoặc np.random.RandomState, tùy chọn): Seed cho bộ sinh số ngẫu nhiên.
        """
        super().__init__(benchmark_func, max_iter, random_state)
        self.step_size = step_size
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self._population_history = []

    def solve(self):
        """
        Thực thi thuật toán Simulated Annealing.

        Returns:
            Giá trị nghiệm tìm được và giá trị hàm mục tiêu tương ứng.
        """
        # Tạo nghiệm ban đầu ngẫu nhiên
        current_solution = self._random.uniform(self.lower_bound, self.upper_bound)
        current_fitness = self.benchmark_func.evaluate(current_solution)

        self._best_solution = current_solution
        self._best_fitness = current_fitness

        temp = self.initial_temp

        for _ in range(self.max_iter):
            # Tạo nghiệm láng giềng ngẫu nhiên
            candidate = current_solution + self._random.uniform(-self.step_size, self.step_size, self.dim)
            candidate = np.clip(candidate, self.lower_bound, self.upper_bound)
            candidate_fitness = self.benchmark_func.evaluate(candidate)

            # Kiểm tra nghiệm mới có tốt hơn không
            if candidate_fitness < self._best_fitness:
                self._best_solution = candidate
                self._best_fitness = candidate_fitness

            diff = candidate_fitness - current_fitness

            if diff < 0 or self._random.rand() < math.exp(-diff / temp):
                current_solution = candidate
                current_fitness = candidate_fitness

            # Làm nguội nhiệt độ
            temp *= self.cooling_rate

            # Lưu lịch sử tối ưu hóa (quá trình hội tụ)
            self._convergence_history.append(self._best_fitness)
            self._solution_history.append(self._best_solution)
            self._population_history.append(np.array([self._best_solution]))

        return self._best_solution, self._best_fitness
