import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class ConvergenceVisualizer:
    """
    Class để visualize quá trình hội tụ của thuật toán optimization.

    Attributes:
        optimizer: Đối tượng optimizer đã chạy solve()
        benchmark_func: Hàm benchmark
        dim: Số chiều của bài toán
        resolution: Độ phân giải cho contour plot
        history: Lịch sử population qua các iteration
        n_iter: Số iteration đã chạy
    """

    def __init__(self, optimizer, resolution=100):
        """
        Khởi tạo visualizer từ optimizer (tương tự CuckooSearch.__init__).

        Args:
            optimizer: Optimizer object
            resolution (int): Số điểm trên mỗi chiều cho grid

        Raises:
            ValueError: Nếu optimizer chưa chạy hoặc không phải 2D
        """
        # Lấy thông tin từ optimizer (giống CuckooSearch)
        self.optimizer = optimizer
        self.benchmark_func = optimizer.benchmark_func
        self.dim = optimizer.dim
        self.resolution = resolution

        # Kiểm tra optimizer
        if not hasattr(optimizer, 'population_history') or not optimizer.population_history:
            raise ValueError("Optimizer chưa chạy hoặc không lưu population_history!")

        # Kiểm tra dimension
        if self.dim != 2:
            raise ValueError("Chỉ hỗ trợ visualization cho 2D!")

        # Lấy history và bounds
        self.history = optimizer.population_history
        self.n_iter = len(self.history)

        self.lower_bounds, self.upper_bounds = self.benchmark_func.get_bounds()
        self.x_min, self.x_max = self.lower_bounds[0], self.upper_bounds[0]
        self.y_min, self.y_max = self.lower_bounds[1], self.upper_bounds[1]

        # Tạo grid cho contour
        self._create_grid()

    def _create_grid(self):
        """Tạo grid và tính giá trị fitness cho contour plot."""
        x = np.linspace(self.x_min, self.x_max, self.resolution)
        y = np.linspace(self.y_min, self.y_max, self.resolution)
        self.X, self.Y = np.meshgrid(x, y)

        # Tính Z values
        self.Z = np.zeros_like(self.X)
        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[1]):
                self.Z[i, j] = self.benchmark_func.evaluate(
                    np.array([self.X[i, j], self.Y[i, j]])
                )

    def _draw_star_spikes(self, ax, x, y, z, color='black', n_spikes=8, spike_length=0.5, linewidth=1.0, alpha=0.5):
        """
        Vẽ chùm tia xoè ra từ một điểm (tối ưu) cho visualize 3D.

        Args:
            ax: matplotlib 3D axis
            x, y, z: tọa độ điểm trung tâm
            color: màu của các tia
            n_spikes: số lượng tia xoay quanh (mặc định 8)
            spike_length: độ dài tia
            linewidth: độ dày tia
            alpha: độ trong suốt

        Returns:
            list of line objects
        """
        lines = []
        angles = np.linspace(0, 2 * np.pi, n_spikes, endpoint=False)

        # Vẽ các tia trong mặt phẳng XY
        for angle in angles:
            dx = spike_length * np.cos(angle)
            dy = spike_length * np.sin(angle)

            line = ax.plot([x, x + dx], [y, y + dy], [z, z],
                           color=color, linewidth=linewidth, alpha=alpha)
            lines.extend(line)

        # Thêm tia lên trên
        line_up = ax.plot([x, x], [y, y], [z, z + spike_length],
                          color=color, linewidth=linewidth, alpha=alpha)
        lines.extend(line_up)

        # Thêm tia xuống dưới
        line_down = ax.plot([x, x], [y, y], [z, z - spike_length],
                            color=color, linewidth=linewidth, alpha=alpha)
        lines.extend(line_down)

        return lines

    def animate_swarm(self, fps=10, save_file=None):
        """
        Tạo animation hiển thị quá trình hội tụ của swarm với matplotlib FuncAnimation.

        Args:
            fps (int): Frames per second
            save_file (str, optional): Đường dẫn file để lưu. None = không lưu

        Returns:
            FuncAnimation object
        """
        # Tạo figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Vẽ contour nền
        contour = ax.contourf(self.X, self.Y, self.Z, levels=50, cmap='jet')
        ax.contour(self.X, self.Y, self.Z, levels=15, colors='black', alpha=0.3, linewidths=0.5)
        fig.colorbar(contour, ax=ax, label='Fitness Value')

        # Khởi tạo scatter plots
        population_scatter = ax.scatter([], [], c='red', s=50, alpha=0.7,
                                       edgecolors='black', linewidths=1.5, label='Population')
        best_scatter = ax.scatter([], [], c='yellow', s=200, marker='*',
                                 edgecolors='black', linewidths=2, label='Best Solution')

        # Text hiển thị iteration
        iteration_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                                fontsize=12, verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Thiết lập axes
        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)
        ax.set_xlabel('X-coordinate', fontsize=12)
        ax.set_ylabel('Y-coordinate', fontsize=12)
        ax.set_title(f'{self.optimizer.__class__.__name__} - Swarm Convergence', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        def init():
            population_scatter.set_offsets(np.empty((0, 2)))
            best_scatter.set_offsets(np.empty((0, 2)))
            iteration_text.set_text('')
            return population_scatter, best_scatter, iteration_text

        def update(frame):
            # Lấy population tại iteration hiện tại
            pop = self.history[frame]
            population_scatter.set_offsets(pop)

            # Tìm best solution
            fitness_values = np.array([self.benchmark_func.evaluate(ind) for ind in pop])
            best_idx = np.argmin(fitness_values)
            best_point = pop[best_idx]
            best_scatter.set_offsets([best_point])

            # Cập nhật text
            iteration_text.set_text(
                f'Iteration: {frame + 1}/{self.n_iter}\n'
                f'Best Fitness: {fitness_values[best_idx]:.4f}'
            )

            return population_scatter, best_scatter, iteration_text

        # Tạo animation
        anim = FuncAnimation(fig, update, init_func=init, frames=self.n_iter,
                            interval=100, blit=True, repeat=True)

        if save_file:
            anim.save(save_file, writer='pillow', fps=fps)
            print(f"Đã lưu animation vào {save_file}")

        plt.tight_layout()
        plt.show()

        return anim

    def plot_trajectories(self, n_samples=10, save_file=None):
        """
        Vẽ quỹ đạo của một số cá thể ngẫu nhiên.

        Args:
            n_samples (int): Số cá thể để vẽ quỹ đạo
            save_file (str, optional): Đường dẫn file để lưu. None = không lưu
        """
        pop_size = self.history[0].shape[0]
        sample_indices = np.random.choice(pop_size, min(n_samples, pop_size), replace=False)

        fig, ax = plt.subplots(figsize=(10, 8))

        # Vẽ contour nền
        contour = ax.contourf(self.X, self.Y, self.Z, levels=50, cmap='jet')
        ax.contour(self.X, self.Y, self.Z, levels=15, colors='black', alpha=0.3, linewidths=0.5)
        fig.colorbar(contour, ax=ax, label='Fitness Value')

        # Vẽ quỹ đạo
        colors = plt.cm.rainbow(np.linspace(0, 1, len(sample_indices)))

        for idx, color in zip(sample_indices, colors):
            trajectory = np.array([self.history[t][idx] for t in range(self.n_iter)])

            # Vẽ đường đi
            ax.plot(trajectory[:, 0], trajectory[:, 1], 'o-',
                   color=color, alpha=0.6, linewidth=1.5, markersize=4)

            # Điểm bắt đầu (Start)
            ax.scatter(trajectory[0, 0], trajectory[0, 1],
                      c=[color], s=100, marker='s', edgecolors='black', linewidths=1.5)

            # Điểm kết thúc (End)
            ax.scatter(trajectory[-1, 0], trajectory[-1, 1],
                      c=[color], s=100, marker='*', edgecolors='black', linewidths=1.5)

        # Legend
        ax.scatter([], [], c='gray', s=100, marker='s',
                                   edgecolors='black', linewidths=1.5, label='Start')
        ax.scatter([], [], c='gray', s=100, marker='*',
                                 edgecolors='black', linewidths=1.5, label='End')
        ax.legend()

        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)
        ax.set_xlabel('X-coordinate', fontsize=12)
        ax.set_ylabel('Y-coordinate', fontsize=12)
        ax.set_title(
            f'{self.optimizer.__class__.__name__} - Trajectories of {n_samples} Individuals',
            fontsize=14
        )
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_file:
            plt.savefig(save_file, dpi=300, bbox_inches='tight')
            print(f"✓ Đã lưu trajectory plot vào {save_file}")

        plt.show()

    def plot_convergence_curve(self, save_file=None):
        """
        Vẽ biểu đồ hội tụ 2D với contour map và đường đi của best solution.

        Args:
            save_file (str, optional): Đường dẫn file để lưu. None = không lưu
        """
        if not hasattr(self.optimizer, 'solution_history') or not self.optimizer.solution_history:
            print("Optimizer không có solution_history!")
            return

        solutions = np.array(self.optimizer.solution_history)

        fitness_history = np.array([self.benchmark_func.evaluate(sol) for sol in solutions])
        best_idx = int(np.argmin(fitness_history))
        best_val = float(fitness_history[best_idx])

        fig, ax = plt.subplots(figsize=(10, 8))

        # Vẽ contour của benchmark function
        contour = ax.contourf(self.X, self.Y, self.Z, levels=50, cmap='jet')
        ax.contour(self.X, self.Y, self.Z, levels=15, colors='black', alpha=0.3, linewidths=0.5)
        fig.colorbar(contour, ax=ax, label='Fitness Value')

        # Vẽ đường đi của nghiệm best solution qua các iteration
        ax.plot(solutions[:, 0], solutions[:, 1], '-o', markersize=3,
                linewidth=1.5, label='Best Solution', color='black')

        # Đánh dấu điểm bắt đầu
        ax.plot(solutions[0, 0], solutions[0, 1], 'go', markersize=8, label='Start')

        # Đánh dấu điểm kết thúc
        ax.plot(solutions[-1, 0], solutions[-1, 1], 'bo', markersize=8, label='End')

        # Đánh dấu điểm best và thêm label giá trị
        ax.scatter(solutions[best_idx, 0], solutions[best_idx, 1],
                   c='yellow', s=180, marker='*', edgecolors='black', linewidths=1.8,
                   label=f'Best: {best_val:.6g}')

        # Thêm ghi chú và hiển thị giá trị bess, iteration ở góc trên trái
        ax.text(0.02, 0.98,
                f'Best Fitness: {best_val:.6g}\nIteration: {best_idx + 1}/{len(solutions)}',
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

        # Thiết lập axes
        ax.set_title(f'{self.optimizer.__class__.__name__} - Convergence Curve',
                    fontsize=14)
        ax.set_xlabel('X-coordinate', fontsize=12)
        ax.set_ylabel('Y-coordinate', fontsize=12)
        ax.legend()

        plt.tight_layout()

        if save_file:
            plt.savefig(save_file, dpi=300, bbox_inches='tight')
            print(f"Đã lưu biểu đồ hội tụ vào {save_file}")

        plt.show()

    def animate_swarm_3d(self, fps=10, save_file=None, alpha=0.7):
        """
        Animation 3D hiển thị quá trình hội tụ của swarm trên bề mặt benchmark.
        Bề mặt 3D được làm mờ (alpha=0.3) để hiển thị điểm dữ liệu rõ hơn.

        Args:
            fps (int): Frames per second
            save_file (str, optional): Đường dẫn file để lưu GIF
            alpha (float): Độ trong suốt của bề mặt benchmark

        Returns:
            FuncAnimation object
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Vẽ bề mặt benchmark (Mờ với alpha)
        surf = ax.plot_surface(self.X, self.Y, self.Z, cmap='jet',
                               alpha=alpha, edgecolor='none', antialiased=True)
        fig.colorbar(surf, ax=ax, shrink=0.5, label='Fitness Value', pad=0.1)

        # Khởi tạo scatter plots
        population_scatter = ax.scatter([], [], [], c='red', s=50,
                                       depthshade=True, label='Population')
        best_scatter = ax.scatter([], [], [], c='yellow', s=200, marker='*',
                                 depthshade=True, label='Best Solution')

        # Danh sách để lưu spikes (chùm tia)
        spikes_pop = []
        spikes_best = []

        # Thiết lập axes
        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)
        ax.set_zlim(self.Z.min(), self.Z.max())
        ax.set_xlabel('X-coordinate', fontsize=12)
        ax.set_ylabel('Y-coordinate', fontsize=12)
        ax.set_zlabel('Fitness', fontsize=12)
        ax.set_title(f'{self.optimizer.__class__.__name__} - 3D Swarm Convergence', fontsize=14)
        ax.legend()
        ax.view_init(elev=20, azim=45)

        def init():
            population_scatter._offsets3d = ([], [], [])
            best_scatter._offsets3d = ([], [], [])
            return [population_scatter, best_scatter]

        def update(frame):
            # Xóa spikes cũ
            for spike in spikes_pop:
                spike.remove()
            for spike in spikes_best:
                spike.remove()
            spikes_pop.clear()
            spikes_best.clear()

            pop = self.history[frame]
            z_coords = np.array([self.benchmark_func.evaluate(ind) for ind in pop])

            # Tìm best
            best_idx = np.argmin(z_coords)

            # Vẽ spikes cho population (chùm tia đen nhỏ)
            for i in range(len(pop)):
                if i != best_idx:
                    spike_lines = self._draw_star_spikes(
                        ax, pop[i, 0], pop[i, 1], z_coords[i],
                        color='black', n_spikes=6, spike_length=0.3,
                        linewidth=0.5, alpha=0.3
                    )
                    spikes_pop.extend(spike_lines)

            # Vẽ spikes cho best solution (chùm tia vàng to)
            spike_lines_best = self._draw_star_spikes(
                ax, pop[best_idx, 0], pop[best_idx, 1], z_coords[best_idx],
                color='gold', n_spikes=12, spike_length=0.6,
                linewidth=2.0, alpha=0.9
            )
            spikes_best.extend(spike_lines_best)

            # Cập nhật scatter points
            population_scatter._offsets3d = (pop[:, 0], pop[:, 1], z_coords)
            best_scatter._offsets3d = ([pop[best_idx, 0]], [pop[best_idx, 1]], [z_coords[best_idx]])

            # Cập nhật title
            ax.set_title(
                f'{self.optimizer.__class__.__name__} - 3D Swarm Convergence\n'
                f'Iteration: {frame + 1}/{self.n_iter} | Best Fitness: {z_coords[best_idx]:.4f}',
                fontsize=12
            )

            return [population_scatter, best_scatter] + spikes_pop + spikes_best

        anim = FuncAnimation(fig, update, init_func=init, frames=self.n_iter,
                            interval=100, blit=False, repeat=True)

        if save_file:
            anim.save(save_file, writer='pillow', fps=fps)
            print(f"Đã lưu 3D animation vào {save_file}")

        plt.tight_layout()
        plt.show()

        return anim

    def plot_trajectories_3d(self, n_samples=10, save_file=None, alpha=0.7):
        """
        Vẽ quỹ đạo 3D của một số cá thể ngẫu nhiên trên bề mặt benchmark.

        Args:
            n_samples (int): Số cá thể để vẽ quỹ đạo
            save_file (str, optional): Đường dẫn file để lưu
            alpha (float): Độ trong suốt của bề mặt benchmark
        """
        pop_size = self.history[0].shape[0]
        sample_indices = np.random.choice(pop_size, min(n_samples, pop_size), replace=False)

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Vẽ bề mặt benchmark (Mờ với alpha)
        surf = ax.plot_surface(self.X, self.Y, self.Z, cmap='jet',
                               alpha=alpha, edgecolor='none', antialiased=True)
        fig.colorbar(surf, ax=ax, shrink=0.5, label='Fitness Value', pad=0.1)

        # Vẽ quỹ đạo
        colors = plt.cm.rainbow(np.linspace(0, 1, len(sample_indices)))

        for idx, color in zip(sample_indices, colors):
            trajectory = np.array([self.history[t][idx] for t in range(self.n_iter)])
            z_coords = np.array([self.benchmark_func.evaluate(pos) for pos in trajectory])

            # Vẽ đường đi
            ax.plot(trajectory[:, 0], trajectory[:, 1], z_coords,
                   'o-', color=color, alpha=0.7, linewidth=2, markersize=4)

            # Điểm bắt đầu
            ax.scatter(trajectory[0, 0], trajectory[0, 1], z_coords[0],
                      c=[color], s=100, marker='s', edgecolors='black',
                      linewidths=1.5, depthshade=True)
            self._draw_star_spikes(
                ax, trajectory[0, 0], trajectory[0, 1], z_coords[0],
                color='black', n_spikes=6, spike_length=0.25,
                linewidth=0.8, alpha=0.4
            )

            # Điểm kết thúc
            ax.scatter(trajectory[-1, 0], trajectory[-1, 1], z_coords[-1],
                      c=[color], s=150, marker='*', edgecolors='black',
                      linewidths=1.5, depthshade=True)
            self._draw_star_spikes(
                ax, trajectory[-1, 0], trajectory[-1, 1], z_coords[-1],
                color=color, n_spikes=10, spike_length=0.4,
                linewidth=1.2, alpha=0.7
            )

        # Legend
        ax.scatter([], [], [], c='gray', s=100, marker='s',
                  edgecolors='black', linewidths=1.5, label='Start')
        ax.scatter([], [], [], c='gray', s=150, marker='*',
                  edgecolors='black', linewidths=1.5, label='End')
        ax.legend()

        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)
        ax.set_zlim(self.Z.min(), self.Z.max())
        ax.set_xlabel('X-coordinate', fontsize=12)
        ax.set_ylabel('Y-coordinate', fontsize=12)
        ax.set_zlabel('Fitness', fontsize=12)
        ax.set_title(
            f'{self.optimizer.__class__.__name__} - 3D Trajectories of {n_samples} Individuals',
            fontsize=14
        )
        ax.view_init(elev=20, azim=45)

        plt.tight_layout()

        if save_file:
            plt.savefig(save_file, dpi=300, bbox_inches='tight')
            print(f"Đã lưu 3D trajectory plot vào {save_file}")

        plt.show()

    def plot_convergence_curve_3d(self, save_file=None, alpha=0.7):
        """
        Vẽ biểu đồ hội tụ 3D với bề mặt benchmark và đường đi của best solution.

        Args:
            save_file (str, optional): Đường dẫn file để lưu
            alpha (float): Độ trong suốt của bề mặt benchmark
        """
        from mpl_toolkits.mplot3d import Axes3D

        if not hasattr(self.optimizer, 'solution_history') or not self.optimizer.solution_history:
            print("Optimizer không có solution_history!")
            return

        solutions = np.array(self.optimizer.solution_history)
        z_coords = np.array([self.benchmark_func.evaluate(sol) for sol in solutions])

        # --- Tính lịch sử fitness và tìm best ---
        best_idx = int(np.argmin(z_coords))
        best_val = float(z_coords[best_idx])
        # -------------------------------------------------

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Vẽ bề mặt benchmark (Mờ với alpha)
        surf = ax.plot_surface(self.X, self.Y, self.Z, cmap='jet',
                               alpha=alpha, edgecolor='none', antialiased=True)
        fig.colorbar(surf, ax=ax, shrink=0.5, label='Fitness Value', pad=0.1)

        # Vẽ đường đi của best solution
        ax.plot(solutions[:, 0], solutions[:, 1], z_coords,
               '-o', markersize=4, linewidth=2, label='Best Solution Path', color='black')

        # Điểm bắt đầu
        ax.scatter(solutions[0, 0], solutions[0, 1], z_coords[0],
                  c='green', s=150, marker='o', edgecolors='black',
                  linewidths=2, depthshade=True, label='Start')

        # Điểm kết thúc
        ax.scatter(solutions[-1, 0], solutions[-1, 1], z_coords[-1],
                  c='blue', s=150, marker='o', edgecolors='black',
                  linewidths=2, depthshade=True, label='End')

        # Đánh dấu điểm best với marker nổi bật
        ax.scatter(solutions[best_idx, 0], solutions[best_idx, 1], z_coords[best_idx],
                  c='yellow', s=300, marker='*', edgecolors='black',
                  linewidths=2.5, depthshade=True, label=f'Best: {best_val:.6g}')

        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)
        ax.set_zlim(self.Z.min(), self.Z.max())
        ax.set_xlabel('X-coordinate', fontsize=12)
        ax.set_ylabel('Y-coordinate', fontsize=12)
        ax.set_zlabel('Fitness', fontsize=12)

        # Cập nhật title với thông tin best fitness
        ax.set_title(
            f'{self.optimizer.__class__.__name__} - 3D Convergence Curve\n'
            f'Best Fitness: {best_val:.6g} | Iteration: {best_idx + 1}/{len(solutions)}',
            fontsize=14
        )

        ax.legend()
        ax.view_init(elev=20, azim=45)

        plt.tight_layout()

        if save_file:
            plt.savefig(save_file, dpi=300, bbox_inches='tight')
            print(f"Đã lưu 3D convergence curve vào {save_file}")
        plt.show()

    @staticmethod
    def display(file_path):
        """
        Hiển thị file ảnh hoặc GIF trong matplotlib figure.

        Args:
            file_path (str): Đường dẫn file (PNG, JPG, JPEG, WEBP, GIF)
        """

        if not os.path.exists(file_path):
            print(f"File không tồn tại: {file_path}")
            return

        ext = os.path.splitext(file_path)[1].lower()

        # Hiển thị GIF
        if ext == '.gif':
            try:
                # Đọc file GIF
                gif = imageio.mimread(file_path)
                frames = [np.array(frame) for frame in gif]

                fig, ax = plt.subplots(figsize=(10, 8))
                img_display = ax.imshow(frames[0])
                ax.axis('off')
                ax.set_title(f'Display GIF: {os.path.basename(file_path)}', fontsize=14)

                def update(frame):
                    img_display.set_array(frame)
                    return [img_display]

                anim = FuncAnimation(fig, update, frames=frames, interval=100, blit=True, repeat=True)

                plt.tight_layout()
                plt.show()

                print(f"Đã hiển thị GIF: {file_path}")

            except Exception as e:
                print(f"Lỗi khi hiển thị GIF: {e}")
        # Hiển thị ảnh tĩnh
        elif ext in ['.png', '.jpg', '.jpeg', '.webp', '.bmp']:
            try:
                from PIL import Image
                img = Image.open(file_path)

                fig, ax = plt.subplots(figsize=(10, 8))
                ax.imshow(img)
                ax.axis('off')
                ax.set_title(f'Display: {os.path.basename(file_path)}', fontsize=14)

                plt.tight_layout()
                plt.show()

                print(f"Đã hiển thị ảnh: {file_path}")

            except Exception as e:
                print(f"Lỗi khi hiển thị ảnh: {e}")
        else:
            print(f"Định dạng file không được hỗ trợ: {ext}")
            print("Hỗ trợ: .gif, .png, .jpg, .jpeg, .webp, .bmp")
