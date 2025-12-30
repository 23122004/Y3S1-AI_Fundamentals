from modules.cnf_generator import HashiCNF

class BaseSolver:
    def __init__(self, grid):
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.cnf_gen = HashiCNF(grid)
        self.clauses = []
        self.var_map = {}
        self.potential_edges = []
        self.islands = []
        self.num_vars = 0

    def _init_cnf(self):
        self.clauses, self.var_map, self.potential_edges, self.islands = self.cnf_gen.generate_clauses()
        self.num_vars = len(self.var_map)

    def _check_connectivity(self, state):
        """Check if all islands are connected via bridges"""
        adj = {i: [] for i in range(len(self.islands))}

        for e in self.potential_edges:
            var_id = self.var_map[(e['idx'], 1)]
            state_idx = var_id - 1
            if state_idx < len(state) and state[state_idx] == True:
                adj[e['u']].append(e['v'])
                adj[e['v']].append(e['u'])

        if len(self.islands) == 0:
            return True

        visited = set()
        queue = [0]
        visited.add(0)

        while queue:
            curr = queue.pop(0)
            for nxt in adj[curr]:
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)

        return len(visited) == len(self.islands)

    def _check_clause(self, clause, state):
        """
        Check a CNF clause with the given state.
        Returns:
            True: Satisfied
            False: Conflicted (all assigned but not satisfied)
            None: Undetermined (some vars unassigned)
        """
        has_unassigned = False

        for literal in clause:
            var_id = abs(literal)
            state_idx = var_id - 1
            val = state[state_idx]

            if val is None:
                has_unassigned = True
            elif (literal > 0 and val is True) or (literal < 0 and val is False):
                return True

        if has_unassigned:
            return None
        return False

    def _check_all_clauses(self, state):
        """
        Check all CNF clauses with the given state.
        Returns:
            True: All satisfied
            False: Conflict exists
            None: Some clauses undetermined but no conflict
        """
        has_unassigned = False

        for clause in self.clauses:
            result = self._check_clause(clause, state)
            if result is False:
                return False
            if result is None:
                has_unassigned = True

        if has_unassigned:
            return None
        return True

    def _state_to_model(self, state):
        """
        Convert state tuple Boolean to model (list of literals).

        state[i] = True  → literal +(i+1) in model
        state[i] = False → literal -(i+1) in model
        """
        model = []
        for state_idx, val in enumerate(state):
            var_id = state_idx + 1
            if val is True:
                model.append(var_id)
            else:
                model.append(-var_id)
        return model

    def solve(self):
        """
        Solves the puzzle.
        Returns:
            model (list): List of true literals (integer) if solution found, else None
        """
        raise NotImplementedError("Subclasses must implement solve()")

