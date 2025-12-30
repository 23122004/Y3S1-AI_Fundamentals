import itertools

class HashiCNF:
    def __init__(self, grid):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.islands = []
        self.potential_edges = []
        self.var_map = {} 
        self.clauses = []
        
        # Parse grid to find islands
        for r in range(self.rows):
            for c in range(self.cols):
                if grid[r][c] > 0:
                    self.islands.append({'r': r, 'c': c, 'val': grid[r][c], 'id': len(self.islands)})

    def generate_clauses(self):
        # Define variables (potential edges)
        self._find_potential_edges()
        
        # Create CNF constraints
        self._gen_consistency() # If there are 2 bridges -> there is 1 bridge
        self._gen_crossing()    # No crossing bridges
        self._gen_capacity()    # Satisfy island bridge count

        return self.clauses, self.var_map, self.potential_edges, self.islands

    def _find_potential_edges(self):
        n = len(self.islands)
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                u, v = self.islands[i], self.islands[j]
                edge_info = None
                
                # Horizontal
                if u['r'] == v['r']:
                    c_min, c_max = min(u['c'], v['c']), max(u['c'], v['c'])
                    if all(self.grid[u['r']][k] == 0 for k in range(c_min + 1, c_max)):
                        edge_info = {'type': 'h', 'r': u['r'], 'c_min': c_min, 'c_max': c_max}
                # Vertical
                elif u['c'] == v['c']:
                    r_min, r_max = min(u['r'], v['r']), max(u['r'], v['r'])
                    if all(self.grid[k][u['c']] == 0 for k in range(r_min + 1, r_max)):
                        edge_info = {'type': 'v', 'c': u['c'], 'r_min': r_min, 'r_max': r_max}
                
                if edge_info:
                    idx = len(edges)
                    self.var_map[(idx, 1)] = len(self.var_map) + 1
                    self.var_map[(idx, 2)] = len(self.var_map) + 1
                    edge_info.update({'u': i, 'v': j, 'idx': idx})
                    edges.append(edge_info)
        self.potential_edges = edges

    def _gen_consistency(self):
        for idx in range(len(self.potential_edges)):
            self.clauses.append([-self.var_map[(idx, 2)], self.var_map[(idx, 1)]])

    def _gen_crossing(self):
        for i, e1 in enumerate(self.potential_edges):
            for j, e2 in enumerate(self.potential_edges):
                if i >= j: continue
                
                # Determine which is horizontal and which is vertical
                h_edge, v_edge = None, None
                if e1['type'] == 'h' and e2['type'] == 'v':
                    h_edge, v_edge = e1, e2
                elif e1['type'] == 'v' and e2['type'] == 'h':
                    h_edge, v_edge = e2, e1
                
                # If we have one of each, check for crossing
                if h_edge and v_edge:
                    if (h_edge['c_min'] < v_edge['c'] < h_edge['c_max']) and \
                       (v_edge['r_min'] < h_edge['r'] < v_edge['r_max']):
                        self.clauses.append([-self.var_map[(i, 1)], -self.var_map[(j, 1)]])

    def _gen_capacity(self):
        for island in self.islands:
            u_id = island['id']
            K = island['val']
            # Collect all variables connecting to this island
            vars_u = []
            for e in self.potential_edges:
                if e['u'] == u_id or e['v'] == u_id:
                    vars_u.append(self.var_map[(e['idx'], 1)])
                    vars_u.append(self.var_map[(e['idx'], 2)])
            
            N = len(vars_u)
            # At most K
            if K < N:
                for s in itertools.combinations(vars_u, K + 1):
                    self.clauses.append([-x for x in s])
            # At least K
            if K > 0 and (N - K + 1) > 0:
                for s in itertools.combinations(vars_u, N - K + 1):
                    self.clauses.append(list(s))
