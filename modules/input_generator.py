import os
import random

class HashiGenerator:
    def __init__(self, width, height, num_islands):
        self.width = width
        self.height = height
        self.target_islands = num_islands
        self.grid = [[0 for _ in range(width)] for _ in range(height)]
        # used_cells stores coordinates that already have islands or bridges to avoid crossing
        self.used_cells = set() 
        self.islands = [] # Stores coordinates of created islands

    def _is_valid_pos(self, r, c):
        return 0 <= r < self.height and 0 <= c < self.width

    def _get_path_cells(self, r1, c1, r2, c2):
        cells = []
        if r1 == r2: # Horizontal
            c_min, c_max = min(c1, c2), max(c1, c2)
            for c in range(c_min + 1, c_max):
                cells.append((r1, c))
        else: # Vertical
            r_min, r_max = min(r1, r2), max(r1, r2)
            for r in range(r_min + 1, r_max):
                cells.append((r, c1))
        return cells

    def generate(self):
        # Place the first island randomly
        start_r = random.randint(0, self.height - 1)
        start_c = random.randint(0, self.width - 1)
        self.grid[start_r][start_c] = 0
        self.islands.append((start_r, start_c))
        self.used_cells.add((start_r, start_c))

        attempts = 0
        # Loop to expand islands
        while len(self.islands) < self.target_islands and attempts < 1000:
            attempts += 1
            src_r, src_c = random.choice(self.islands)
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            random.shuffle(directions)
            
            found_move = False
            for dr, dc in directions:
                # Random island distance from 2 to min(width, height)
                dist = random.randint(2, max(self.width, self.height) // 2)
                dst_r, dst_c = src_r + dr * dist, src_c + dc * dist
                
                # Check validity
                if not self._is_valid_pos(dst_r, dst_c):
                    continue
                
                if (dst_r, dst_c) in self.used_cells:
                    continue

                path = self._get_path_cells(src_r, src_c, dst_r, dst_c)
                if any(cell in self.used_cells for cell in path):
                    continue

                # --- IF VALID, CONNECT BRIDGE ---
                bridges = random.choice([1, 2]) # 1 or 2 bridges
                if self.grid[src_r][src_c] + bridges > 8: continue
                self.grid[src_r][src_c] += bridges
                
                self.grid[dst_r][dst_c] = bridges 
                
                self.islands.append((dst_r, dst_c))
                self.used_cells.add((dst_r, dst_c))
                for cell in path:
                    self.used_cells.add(cell)
                
                found_move = True
                break
            
            if not found_move:
                pass

        return self.grid

    def print_grid(self):
        print(f"--- Generated Map ({len(self.islands)} islands) ---")
        for row in self.grid:
            print("  ".join([str(x) if x > 0 else "." for x in row]))
            
    def save_to_file(self, filename):
        with open(filename, "w") as f:
            for row in self.grid:
                f.write(", ".join(map(str, row)) + "\n")