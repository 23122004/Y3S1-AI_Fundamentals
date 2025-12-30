from pysat.solvers import Glucose3
from modules.solvers.base_solver import BaseSolver

class PySatSolver(BaseSolver):
    def __init__(self, grid):
        super().__init__(grid)
        self.sat_solver = Glucose3()
        self.disconnected_component = []

    def solve(self):
        # Generate core CNF clauses
        self._init_cnf()
        
        # Add generated clauses to the SAT solver
        for clause in self.clauses:
            self.sat_solver.add_clause(clause)
        
        # Solve loop with connectivity check
        while self.sat_solver.solve():
            model = self.sat_solver.get_model()
            if self._check_connectivity(model):
                return model
            else:
                self._add_connectivity_constraint()
        
        return None

    def _check_connectivity(self, model):
        """
        Check if the graph is connected based on the solution model.
        Returns: True if the graph is connected, False otherwise.
        """
        model_set = set(model)
        adj = {i: [] for i in range(len(self.islands))}
        
        # Build the graph from the solution (only edges with at least 1 bridge)
        for e in self.potential_edges:
            if self.var_map[(e['idx'], 1)] in model_set:
                adj[e['u']].append(e['v'])
                adj[e['v']].append(e['u'])
        
        # BFS to find all connected components
        all_components = []
        remaining = set(range(len(self.islands)))
        
        while remaining:
            # Select any unvisited island
            start = next(iter(remaining))
            visited = set()
            q = [start]
            visited.add(start)
            
            # BFS to find the connected component containing the start island
            while q:
                curr = q.pop(0)
                for nxt in adj[curr]:
                    if nxt not in visited:
                        visited.add(nxt)
                        q.append(nxt)
            
            all_components.append(visited)
            remaining -= visited
        
        # If only 1 component -> graph is connected
        if len(all_components) == 1:
            return True
        
        # If more than 1 component -> save the smallest one to generate cut constraint
        self.disconnected_component = min(all_components, key=len)
        return False

    def _add_connectivity_constraint(self):
        """
        Add a cut constraint to eliminate disconnected solutions.
        """
        S = self.disconnected_component
        cut_clause = []
        
        # Find all edges connecting S and the rest (cut edges)
        for e in self.potential_edges:
            # Check if edge e connects an island in S with an island outside S
            u_in_S = e['u'] in S
            v_in_S = e['v'] in S
            
            # If only one endpoint belongs to S (XOR) -> this is a cut edge
            if u_in_S != v_in_S:
                cut_clause.append(self.var_map[(e['idx'], 1)])
        
        # Add constraint: at least one of the cut edges must have a bridge
        if cut_clause:
            self.sat_solver.add_clause(cut_clause)
