# Hướng dẫn chạy dự án Hashi Solver

Dự án này thực hiện so sánh hiệu năng các thuật toán giải game Hashi.

## Cài đặt môi trường

Cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

## Cách chạy

### 1. Chạy Benchmark

Chạy lệnh sau để sinh bộ test, chạy các thuật toán giải (PySAT, A*, Backtracking, Brute Force) và ghi lại thời gian chạy:

```bash
python benchmark.py
```

*   Các cài đặt về `timeout` và `map sizes` có thể được điều chỉnh trong file `config.py`.
*   Kết quả thời gian chạy sẽ được lưu vào file `output/benchmark.csv`.
*   Lời giải chi tiết sẽ nằm trong thư mục `output/solutions/`.

### 2. Vẽ biểu đồ phân tích

Sau khi chạy xong benchmark, chạy lệnh sau để vẽ biểu đồ so sánh:

```bash
python analyze.py
```

*   Biểu đồ sẽ được hiển thị và lưu tại `output/benchmark_plot.png`.