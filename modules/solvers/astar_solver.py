import sys
import heapq
from modules.solvers.base_solver import BaseSolver

sys.setrecursionlimit(5000)

class AStarSolver(BaseSolver):
    def __init__(self, grid):
        super().__init__(grid)
        self.edge_to_islands = {}  # edge_idx -> (island_u, island_v)
        self.island_to_edges = {}  # island_id -> list of edge_idx
        self.var_to_clauses = {}   # var_id -> list of clause indices
        # Note: var index begins from 1, state index begins from 0

    def solve(self):
        self._init_cnf()

        # Build edge-island mapping
        for i in range(len(self.islands)):
            self.island_to_edges[i] = []

        for e in self.potential_edges:
            edge_idx = e['idx']
            u, v = e['u'], e['v']
            self.edge_to_islands[edge_idx] = (u, v)
            self.island_to_edges[u].append(edge_idx)
            self.island_to_edges[v].append(edge_idx)

        # Pre-index clauses by variable
        for clause_idx, clause in enumerate(self.clauses):
            for literal in clause:
                var_id = abs(literal)
                if var_id not in self.var_to_clauses:
                    self.var_to_clauses[var_id] = []
                self.var_to_clauses[var_id].append(clause_idx)

        # A* search
        result = self._astar_search()

        if result is not None:
            return self._state_to_model(result)

        return None

    def _get_next_var_idx(self, state):
        """
        Chooses the next variable index to assign based on MRV heuristic.
        Prioritizes islands with the highest ratio of (missing bridges / remaining edges).

        Returns: Index of the variable to assign next. None if all assigned.
        """
        num_islands = len(self.islands)
        island_current = [0] * num_islands
        island_remaining_edges = [[] for _ in range(num_islands)]

        for e in self.potential_edges:
            var_1 = self.var_map[(e['idx'], 1)]
            var_2 = self.var_map[(e['idx'], 2)]
            state_idx_1 = var_1 - 1
            state_idx_2 = var_2 - 1

            bridges = 0
            if state_idx_1 < len(state):
                bridges += (state[state_idx_1] == True)
            if state_idx_2 < len(state):
                bridges += (state[state_idx_2] == True)

            if bridges > 0:
                island_current[e['u']] += bridges
                island_current[e['v']] += bridges

            if state[state_idx_1] is None:
                island_remaining_edges[e['u']].append(state_idx_1)
                island_remaining_edges[e['v']].append(state_idx_1)

            if state[state_idx_2] is None:
                island_remaining_edges[e['u']].append(state_idx_2)
                island_remaining_edges[e['v']].append(state_idx_2)

        # Search for the island with the highest ratio
        best_var_idx = None
        best_ratio = -1

        for island in self.islands:
            island_id = island['id']
            needed = island['val']

            current = island_current[island_id]
            missing = needed - current
            remaining = island_remaining_edges[island_id]

            if missing > 0 and len(remaining) > 0:
                ratio = missing / len(remaining)
                if ratio > best_ratio:
                    best_ratio = ratio
                    # Choose the first unassigned variable
                    for var_idx in remaining:
                        if state[var_idx] is None:
                            best_var_idx = var_idx
                            break

        if best_var_idx is None:
            for i, val in enumerate(state):
                if val is None:
                    return i

        return best_var_idx

    def _check_clauses(self, state, var_idx):
        """
        Check only clauses that contain the variable at var_idx.
        """
        var_id = var_idx + 1

        if var_id not in self.var_to_clauses:
            return True

        for clause_idx in self.var_to_clauses[var_id]:
            clause = self.clauses[clause_idx]

            satisfied = False
            all_assigned = True

            for literal in clause:
                var = abs(literal)
                state_idx = var - 1
                val = state[state_idx] if state_idx < len(state) else None

                if val is None:
                    all_assigned = False
                elif (literal > 0 and val is True) or (literal < 0 and val is False):
                    satisfied = True
                    break

            if all_assigned and not satisfied:
                return False

        return True

    def _astar_search(self):
        initial_state = tuple([None] * self.num_vars)

        # Priority queue: (f_score, counter, state)
        # counters to tie-break between states with same f_score
        counter = 0
        open_set = []
        closed_set = set()
        heapq.heappush(open_set, (self._compute_priority(initial_state), counter, initial_state))

        while open_set:
            f_score, _, current_state = heapq.heappop(open_set)

            if current_state in closed_set:
                continue
            closed_set.add(current_state)

            # Check connectivity if all variables are assigned
            if all(b is not None for b in current_state):
                if self._check_connectivity(current_state):
                    return current_state
                continue

            # Choose the next variable index to assign
            var_idx = self._get_next_var_idx(current_state)
            if var_idx is None:
                continue

            for val in [True, False]:
                new_state = list(current_state)
                new_state[var_idx] = val
                new_state = tuple(new_state)

                if new_state not in closed_set:
                    # Check clauses involving var_idx
                    if self._check_clauses(new_state, var_idx):
                        counter += 1
                        priority = self._compute_priority(new_state)
                        heapq.heappush(open_set, (priority, counter, new_state))

        return None

    def _compute_cost(self, state):
        total_bridges = 0
        for e in self.potential_edges:
            var_1 = self.var_map[(e['idx'], 1)]
            var_2 = self.var_map[(e['idx'], 2)]
            state_idx_1 = var_1 - 1
            state_idx_2 = var_2 - 1

            if state_idx_1 < len(state):
                total_bridges += (state[state_idx_1] == True)
            if state_idx_2 < len(state):
                total_bridges += (state[state_idx_2] == True)

        return total_bridges

    def _compute_heuristic(self, state):
        island_bridges = [0] * len(self.islands)
        unassigned_edges = 0

        for e in self.potential_edges:
            var_1 = self.var_map[(e['idx'], 1)]
            var_2 = self.var_map[(e['idx'], 2)]
            state_idx_1 = var_1 - 1
            state_idx_2 = var_2 - 1

            # Count unassigned edges
            if state[state_idx_1] is None:
                unassigned_edges += 1

            bridges = 0
            if state_idx_1 < len(state):
                bridges += (state[state_idx_1] == True)
            if state_idx_2 < len(state):
                bridges += (state[state_idx_2] == True)

            if bridges > 0:
                island_bridges[e['u']] += bridges
                island_bridges[e['v']] += bridges

        # Total missing of bridges needed
        total_missing = 0
        for island in self.islands:
            needed = island['val']
            current = island_bridges[island['id']]
            if current < needed:
                total_missing += (needed - current)

        if unassigned_edges > 0:
            return total_missing / unassigned_edges
        return 0

    def _compute_priority(self, state):
        """
        Compute f(n) = g(n) + w * h(n)

        g(n): Total number of bridges built
              Range: [0, 2 * len(self.potential_edges)]

        h(n): Ratio of missing bridges to unassigned edges
              Range: [0, ~2]

        w: Weight factor to balance range of h has similar range as g
        """
        g = self._compute_cost(state)
        h = self._compute_heuristic(state)
        w = len(self.potential_edges)

        return g + w * h

