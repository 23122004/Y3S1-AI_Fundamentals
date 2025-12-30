from config import *

def visualize(optimizer, alpha=0.7):
    algo_name = optimizer.__class__.__name__

    print("\n" + "=" * 60)
    print(f"--- VISUALIZE {algo_name} ---")

    print("\nChạy thuật toán")
    best_solution, best_fitness = optimizer.solve()
    print(f"   ✓ Best fitness: {best_fitness:.8f}")

    print("\nKhởi tạo ConvergenceVisualizer")
    visualizer = ConvergenceVisualizer(optimizer, resolution=50)
    print(f"   ✓ Đã load {visualizer.n_iter} iterations")
    print(f"   ✓ Population size: {visualizer.history[0].shape[0]}")
    print(f"   ✓ Bounds: [{visualizer.x_min:.1f}, {visualizer.x_max:.1f}]")

    print("\n" + "-" * 60)
    print("CÁC VISUALIZATION METHODS:")
    print("-" * 60)

    # Method 1: Animation 2D với matplotlib
    print("\n[1] animate_swarm() - FuncAnimation")
    print("    → Hiển thị animation trực tiếp + lưu GIF")
    visualizer.animate_swarm(save_file=f"{VISUAL_DIR}/{algo_name.lower()}_animation_2d.gif")

    # Method 2: Trajectories 2D
    print("\n[2] plot_trajectories() - Quỹ đạo cá thể")
    print("    → Vẽ đường đi của 5 cá thể")
    visualizer.plot_trajectories(n_samples=5, save_file=f"{VISUAL_DIR}/{algo_name.lower()}_trajectories_2d.png")

    # Method 3: Convergence Curve 2D
    print("\n[3] plot_convergence_curve() - Biểu đồ hội tụ")
    print("    → Hiển thị biểu đồ đường hội tụ")
    visualizer.plot_convergence_curve(save_file=f"{VISUAL_DIR}/{algo_name.lower()}_convergence_curve_2d.png")

    # Method 4: Display GIF (static method)
    print("\n[4] display() - Hiển thị Ảnh/GIF đã lưu")
    print("    → Xem lại Ảnh/GIF trong matplotlib")
    ConvergenceVisualizer.display(f'{VISUAL_DIR}/{algo_name.lower()}_animation_2d.gif')

    # Method 5: Animation 3D với matplotlib
    print("\n[5] animate_swarm_3d() - FuncAnimation")
    print("    → Hiển thị animation trực tiếp + lưu GIF")
    visualizer.animate_swarm_3d(save_file=f"{VISUAL_DIR}/{algo_name.lower()}_animation_3d.gif", alpha=alpha)

    # Method 6: Trajectories 3D
    print("\n[6] plot_trajectories_3d() - Quỹ đạo cá thể")
    print("    → Vẽ đường đi của 5 cá thể")
    visualizer.plot_trajectories_3d(n_samples=5, save_file=f"{VISUAL_DIR}/{algo_name.lower()}_trajectories_3d.png", alpha=alpha)

    # Method 7: Convergence Curve 3D
    print("\n[7] plot_convergence_curve_3d() - Biểu đồ hội tụ")
    print("    → Hiển thị biểu đồ đường hội tụ")
    visualizer.plot_convergence_curve_3d(save_file=f"{VISUAL_DIR}/{algo_name.lower()}_convergence_curve_3d.png", alpha=alpha)

    print("\n" + "-" * 60)
    print("✓ HOÀN THÀNH!")
    print("-" * 60)
    print("\nĐã tạo các file:")
    print(f"  📁 {algo_name.lower()}_animation_2d.gif")
    print(f"  📁 {algo_name.lower()}_trajectories_2d.png")
    print(f"  📁 {algo_name.lower()}_convergence_curve_2d.png")
    print(f"  📁 {algo_name.lower()}_animation_3d.gif")
    print(f"  📁 {algo_name.lower()}_trajectories_3d.png")
    print(f"  📁 {algo_name.lower()}_convergence_curve_3d.png")

def main(benchmark_name):
    benchmark = None
    if benchmark_name == "Sphere":
        benchmark = Sphere(dim=2)
    elif benchmark_name == "Rastrigin":
        benchmark = Rastrigin(dim=2)

    hc_sa_step_size = {
        "Sphere":  [5.0, 5.0],
        "Rastrigin": [0.2, 1.0]
    }

    cs_optimizer = CuckooSearch(
        benchmark_func=benchmark,
        max_iter=100,
        population_size=40,
        random_state=42,
        alpha=0.1,
        pa=0.5
    )

    fa_optimizer = FireflyAlgorithm(
        benchmark_func=benchmark,
        max_iter=100,
        population_size=40,
        random_state=42,
        alpha=0.1,
        beta0=2.0,
        gamma=0.1
    )

    abc_optimizer = ArtificialBeeColony(
        benchmark_func=benchmark,
        max_iter=100,
        population_size=40,
        random_state=42,
        limit=40
    )

    hc_optimizer = HillClimbing(
        benchmark_func=benchmark,
        max_iter=300,
        step_size=hc_sa_step_size[benchmark_name][0],
        random_state=42
    )

    sa_optimizer = SimulatedAnnealing(
        benchmark_func=benchmark,
        max_iter=300,
        step_size=hc_sa_step_size[benchmark_name][1],
        initial_temp=500.0,
        cooling_rate=0.98,
        random_state=42
    )

    alpha = 0.3
    visualize(cs_optimizer, alpha=alpha)
    visualize(fa_optimizer, alpha=alpha)
    visualize(abc_optimizer, alpha=alpha)
    visualize(hc_optimizer, alpha=alpha)
    visualize(sa_optimizer, alpha=alpha)

if __name__ == "__main__":
    main("Rastrigin")
