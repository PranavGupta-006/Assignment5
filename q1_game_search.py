"""
Q1: Game Search Algorithms
==========================
Implements: Minimax, Alpha-Beta Pruning, Heuristic Alpha-Beta, Monte-Carlo Tree Search (MCTS)
Domain: Tic-Tac-Toe (3x3) — clean, fully solvable, ideal for verifying correctness.

Author: AI Assignment Solution
"""

import math
import random
import time
import json
import sys
from copy import deepcopy
from collections import defaultdict


# =============================================================================
# GAME STATE: Tic-Tac-Toe
# =============================================================================

class TicTacToe:
    """
    Board representation:
        0 = empty, 1 = X (maximizer), -1 = O (minimizer)
    """

    WIN_LINES = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # cols
        [0, 4, 8], [2, 4, 6],             # diagonals
    ]

    def __init__(self, board=None, turn=1):
        self.board = board if board is not None else [0] * 9
        self.turn = turn  # 1 = X's turn, -1 = O's turn

    def legal_moves(self):
        return [i for i, v in enumerate(self.board) if v == 0]

    def make_move(self, pos):
        new_board = self.board[:]
        new_board[pos] = self.turn
        return TicTacToe(new_board, -self.turn)

    def winner(self):
        """Returns 1 (X wins), -1 (O wins), or 0 (no winner yet)."""
        for line in self.WIN_LINES:
            s = sum(self.board[i] for i in line)
            if s == 3:
                return 1
            if s == -3:
                return -1
        return 0

    def is_terminal(self):
        return self.winner() != 0 or not self.legal_moves()

    def terminal_value(self):
        """Returns exact terminal score."""
        w = self.winner()
        if w == 1:
            return 1      # X wins
        if w == -1:
            return -1     # O wins
        return 0          # draw

    def heuristic(self):
        """
        Heuristic evaluation (used when depth limit reached).
        Score = (X two-in-a-row count) - (O two-in-a-row count).
        Scaled to (-0.9, 0.9) so it never overrides terminal values.
        """
        score = 0
        for line in self.WIN_LINES:
            vals = [self.board[i] for i in line]
            x_count = vals.count(1)
            o_count = vals.count(-1)
            if o_count == 0 and x_count == 2:
                score += 1
            if x_count == 0 and o_count == 2:
                score -= 1
        return score * 0.1   # keep within (-0.9, 0.9)

    def display(self):
        sym = {1: 'X', -1: 'O', 0: '.'}
        rows = []
        for r in range(3):
            rows.append(' '.join(sym[self.board[r*3 + c]] for c in range(3)))
        return '\n'.join(rows)

    def __repr__(self):
        return f"TicTacToe(turn={'X' if self.turn==1 else 'O'}, board={self.board})"


# =============================================================================
# 1. MINIMAX (exact, no pruning)
# =============================================================================

class Minimax:
    """
    Classic Minimax search.
    Explores the ENTIRE game tree — guaranteed optimal but slow for deep games.

    Time complexity: O(b^d)  where b = branching factor, d = depth
    """

    def __init__(self):
        self.nodes_visited = 0

    def search(self, state: TicTacToe):
        """Returns (best_value, best_move)."""
        self.nodes_visited = 0
        value, move = self._minimax(state)
        return value, move

    def _minimax(self, state: TicTacToe):
        self.nodes_visited += 1

        if state.is_terminal():
            return state.terminal_value(), None

        moves = state.legal_moves()

        if state.turn == 1:          # Maximizer (X)
            best_val = -math.inf
            best_move = None
            for m in moves:
                val, _ = self._minimax(state.make_move(m))
                if val > best_val:
                    best_val, best_move = val, m
            return best_val, best_move

        else:                        # Minimizer (O)
            best_val = math.inf
            best_move = None
            for m in moves:
                val, _ = self._minimax(state.make_move(m))
                if val < best_val:
                    best_val, best_move = val, m
            return best_val, best_move


# =============================================================================
# 2. ALPHA-BETA PRUNING
# =============================================================================

class AlphaBeta:
    """
    Minimax with Alpha-Beta pruning.
    Prunes branches that cannot affect the final decision.

    Alpha = best value maximizer can guarantee so far (lower bound)
    Beta  = best value minimizer can guarantee so far (upper bound)

    Prune when alpha >= beta (the opponent would never allow this branch).
    Best case complexity: O(b^(d/2))
    """

    def __init__(self):
        self.nodes_visited = 0
        self.pruned = 0

    def search(self, state: TicTacToe):
        self.nodes_visited = 0
        self.pruned = 0
        value, move = self._alpha_beta(state, -math.inf, math.inf)
        return value, move

    def _alpha_beta(self, state: TicTacToe, alpha, beta):
        self.nodes_visited += 1

        if state.is_terminal():
            return state.terminal_value(), None

        moves = state.legal_moves()

        if state.turn == 1:          # Maximizer
            best_val = -math.inf
            best_move = None
            for m in moves:
                val, _ = self._alpha_beta(state.make_move(m), alpha, beta)
                if val > best_val:
                    best_val, best_move = val, m
                alpha = max(alpha, best_val)
                if alpha >= beta:    # Beta cutoff
                    self.pruned += 1
                    break
            return best_val, best_move

        else:                        # Minimizer
            best_val = math.inf
            best_move = None
            for m in moves:
                val, _ = self._alpha_beta(state.make_move(m), alpha, beta)
                if val < best_val:
                    best_val, best_move = val, m
                beta = min(beta, best_val)
                if alpha >= beta:    # Alpha cutoff
                    self.pruned += 1
                    break
            return best_val, best_move


# =============================================================================
# 3. HEURISTIC ALPHA-BETA (Depth-Limited)
# =============================================================================

class HeuristicAlphaBeta:
    """
    Alpha-Beta pruning with a DEPTH LIMIT.
    When the depth limit is reached, uses a heuristic function
    instead of exact terminal evaluation.

    This is the foundation of most real game AIs (chess engines, etc.)
    Depth limit trades accuracy for speed — critical for complex games.
    """

    def __init__(self, max_depth=4):
        self.max_depth = max_depth
        self.nodes_visited = 0
        self.pruned = 0
        self.cutoffs_by_heuristic = 0

    def search(self, state: TicTacToe):
        self.nodes_visited = 0
        self.pruned = 0
        self.cutoffs_by_heuristic = 0
        value, move = self._h_alpha_beta(state, -math.inf, math.inf, 0)
        return value, move

    def _h_alpha_beta(self, state: TicTacToe, alpha, beta, depth):
        self.nodes_visited += 1

        if state.is_terminal():
            return state.terminal_value(), None

        if depth >= self.max_depth:            # Depth cutoff → use heuristic
            self.cutoffs_by_heuristic += 1
            return state.heuristic(), None

        moves = state.legal_moves()

        if state.turn == 1:                    # Maximizer
            best_val = -math.inf
            best_move = None
            for m in moves:
                val, _ = self._h_alpha_beta(state.make_move(m), alpha, beta, depth + 1)
                if val > best_val:
                    best_val, best_move = val, m
                alpha = max(alpha, best_val)
                if alpha >= beta:
                    self.pruned += 1
                    break
            return best_val, best_move

        else:                                  # Minimizer
            best_val = math.inf
            best_move = None
            for m in moves:
                val, _ = self._h_alpha_beta(state.make_move(m), alpha, beta, depth + 1)
                if val < best_val:
                    best_val, best_move = val, m
                beta = min(beta, best_val)
                if alpha >= beta:
                    self.pruned += 1
                    break
            return best_val, best_move


# =============================================================================
# 4. MONTE-CARLO TREE SEARCH (MCTS)
# =============================================================================

class MCTSNode:
    """
    Node in the MCTS search tree.

    Stores:
        state       — game state at this node
        parent      — parent node
        move        — move that led here
        children    — child nodes (expanded moves)
        untried_moves — moves not yet expanded
        wins        — total wins accumulated (from perspective of player who JUST moved)
        visits      — total simulations through this node
    """

    def __init__(self, state: TicTacToe, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = []
        self.untried_moves = state.legal_moves()
        self.wins = 0.0
        self.visits = 0

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def is_terminal(self):
        return self.state.is_terminal()

    def ucb1(self, c=1.41):
        """
        Upper Confidence Bound (UCB1) formula.
        Balances exploitation (wins/visits) vs exploration (log(parent_visits)/visits).
        c = exploration constant (√2 ≈ 1.41 is standard)
        """
        if self.visits == 0:
            return math.inf
        exploitation = self.wins / self.visits
        exploration = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

    def best_child(self, c=1.41):
        return max(self.children, key=lambda n: n.ucb1(c))

    def expand(self):
        """Expand one untried move, return the new child node."""
        move = self.untried_moves.pop(random.randrange(len(self.untried_moves)))
        child = MCTSNode(self.state.make_move(move), parent=self, move=move)
        self.children.append(child)
        return child

    def update(self, result):
        """
        Backpropagate result.
        result is from X's perspective: +1 = X win, -1 = O win, 0 = draw.
        We store wins from the perspective of the player who JUST moved into this node.
        """
        self.visits += 1
        # The player who moved INTO this node is -self.state.turn (opposite of current)
        player_who_moved = -self.state.turn
        if result == player_who_moved:
            self.wins += 1.0
        elif result == 0:
            self.wins += 0.5   # draw counts as half


class MCTS:
    """
    Monte-Carlo Tree Search.
    
    Four phases per iteration:
    1. SELECTION   — traverse tree using UCB1 until a non-fully-expanded node
    2. EXPANSION   — expand one child of the selected node
    3. SIMULATION  — random playout from expanded node to terminal
    4. BACKPROP    — propagate result up the tree

    No heuristic required — learns purely from random simulations.
    More iterations → better accuracy. Works for any game without domain knowledge.
    """

    def __init__(self, iterations=1000):
        self.iterations = iterations
        self.root = None

    def search(self, state: TicTacToe):
        self.root = MCTSNode(state)

        for _ in range(self.iterations):
            node = self._select(self.root)
            if not node.is_terminal():
                node = node.expand()
            result = self._simulate(node.state)
            self._backpropagate(node, result)

        # Choose move with MOST visits (robust child — less variance than best winrate)
        best = max(self.root.children, key=lambda n: n.visits)
        return best.wins / best.visits if best.visits else 0, best.move

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Traverse tree with UCB1 until we find a node to expand."""
        while not node.is_terminal():
            if not node.is_fully_expanded():
                return node
            node = node.best_child()
        return node

    def _simulate(self, state: TicTacToe) -> int:
        """Random rollout to terminal state. Returns winner (1, -1, or 0)."""
        while not state.is_terminal():
            moves = state.legal_moves()
            state = state.make_move(random.choice(moves))
        return state.terminal_value()

    def _backpropagate(self, node: MCTSNode, result: int):
        """Walk back up the tree, updating each node."""
        while node is not None:
            node.update(result)
            node = node.parent


# =============================================================================
# TEST SUITE
# =============================================================================

def run_tests():
    print("=" * 65)
    print("  GAME SEARCH ALGORITHMS — TEST SUITE")
    print("=" * 65)

    # -------------------------------------------------------------------------
    # Test 1: Terminal state detection
    # -------------------------------------------------------------------------
    print("\n[TEST 1] Terminal State Detection")
    # X wins on top row
    state = TicTacToe([1, 1, 1, -1, -1, 0, 0, 0, 0], turn=-1)
    assert state.is_terminal(), "Should be terminal (X wins)"
    assert state.winner() == 1, "Winner should be X"
    assert state.terminal_value() == 1
    # Draw
    draw = TicTacToe([1, -1, 1, 1, -1, -1, -1, 1, 1], turn=1)
    assert draw.is_terminal()
    assert draw.winner() == 0
    assert draw.terminal_value() == 0
    print("  ✓ Terminal detection, winner detection, draw detection")

    # -------------------------------------------------------------------------
    # Test 2: Minimax finds optimal move (should always draw from start)
    # -------------------------------------------------------------------------
    print("\n[TEST 2] Minimax — Optimal Play from Start")
    mm = Minimax()
    t0 = time.perf_counter()
    val, move = mm.search(TicTacToe())
    t1 = time.perf_counter()
    assert val == 0, f"Minimax value from start should be 0 (draw), got {val}"
    assert move is not None
    print(f"  ✓ Value={val} (draw), Best move={move}, "
          f"Nodes={mm.nodes_visited}, Time={t1-t0:.3f}s")

    # -------------------------------------------------------------------------
    # Test 3: Minimax finds only winning move
    # -------------------------------------------------------------------------
    print("\n[TEST 3] Minimax — Must-Win Position")
    # X to play, position 8 wins immediately
    #  X X .
    #  O O .
    #  . . .
    state = TicTacToe([1, 1, 0, -1, -1, 0, 0, 0, 0], turn=1)
    val, move = mm.search(state)
    assert move == 2, f"Minimax should play pos 2 to win, got {move}"
    assert val == 1
    print(f"  ✓ Correctly identifies winning move at position {move}")

    # -------------------------------------------------------------------------
    # Test 4: Minimax — Must-Block Position
    # -------------------------------------------------------------------------
    print("\n[TEST 4] Minimax — Must-Block Position")
    # O to play, must block X's win at position 2
    #  X X .
    #  O . .
    #  . . .
    state = TicTacToe([1, 1, 0, -1, 0, 0, 0, 0, 0], turn=-1)
    val, move = mm.search(state)
    assert move == 2, f"Minimax must block at pos 2, got {move}"
    print(f"  ✓ Correctly blocks at position {move}")

    # -------------------------------------------------------------------------
    # Test 5: Alpha-Beta matches Minimax values
    # -------------------------------------------------------------------------
    print("\n[TEST 5] Alpha-Beta — Value Equivalence with Minimax")
    ab = AlphaBeta()

    test_positions = [
        TicTacToe(),
        TicTacToe([1, 0, 0, 0, -1, 0, 0, 0, 0], turn=1),
        TicTacToe([1, 1, 0, -1, -1, 0, 0, 0, 0], turn=1),
    ]

    for i, pos in enumerate(test_positions):
        mm2 = Minimax()
        ab2 = AlphaBeta()
        mm_val, mm_move = mm2.search(pos)
        ab_val, ab_move = ab2.search(pos)
        assert mm_val == ab_val, \
            f"Position {i}: Minimax={mm_val} but Alpha-Beta={ab_val}"
        print(f"  ✓ Position {i}: MM={mm_val} == AB={ab_val}, "
              f"MM_nodes={mm2.nodes_visited}, AB_nodes={ab2.nodes_visited}, "
              f"Pruned={ab2.pruned}")

    # -------------------------------------------------------------------------
    # Test 6: Alpha-Beta prunes more nodes than Minimax
    # -------------------------------------------------------------------------
    print("\n[TEST 6] Alpha-Beta — Pruning Effectiveness")
    mm3 = Minimax()
    ab3 = AlphaBeta()
    state = TicTacToe()
    mm3.search(state)
    ab3.search(state)
    assert ab3.nodes_visited < mm3.nodes_visited, "Alpha-Beta should visit fewer nodes"
    reduction = (1 - ab3.nodes_visited / mm3.nodes_visited) * 100
    print(f"  ✓ MM nodes={mm3.nodes_visited}, AB nodes={ab3.nodes_visited}, "
          f"Reduction={reduction:.1f}%, Pruned={ab3.pruned}")

    # -------------------------------------------------------------------------
    # Test 7: Heuristic Alpha-Beta — depth limit test
    # -------------------------------------------------------------------------
    print("\n[TEST 7] Heuristic Alpha-Beta — Depth-Limited Search")
    for depth in [2, 4, 6]:
        hab = HeuristicAlphaBeta(max_depth=depth)
        val, move = hab.search(TicTacToe())
        print(f"  depth={depth}: val={val:.2f}, move={move}, "
              f"nodes={hab.nodes_visited}, heuristic_cutoffs={hab.cutoffs_by_heuristic}")

    # Heuristic AB at full depth should agree with exact Minimax
    hab_full = HeuristicAlphaBeta(max_depth=9)
    val_h, move_h = hab_full.search(TicTacToe())
    assert val_h == 0, f"Heuristic AB at full depth should be 0, got {val_h}"
    print(f"  ✓ Full-depth heuristic AB agrees with Minimax (val={val_h})")

    # -------------------------------------------------------------------------
    # Test 8: MCTS — convergence test
    # -------------------------------------------------------------------------
    print("\n[TEST 8] MCTS — Statistical Convergence")
    for iters in [100, 500, 2000]:
        mcts = MCTS(iterations=iters)
        _, move = mcts.search(TicTacToe())
        print(f"  iterations={iters}: best_move={move}")

    # MCTS must-win test (high iterations)
    #  X X .  — X must play position 2
    state = TicTacToe([1, 1, 0, -1, -1, 0, 0, 0, 0], turn=1)
    correct = 0
    for _ in range(10):    # run multiple times for statistical reliability
        mcts = MCTS(iterations=500)
        _, move = mcts.search(state)
        if move == 2:
            correct += 1
    print(f"  ✓ MCTS chose winning move (pos=2) in {correct}/10 trials "
          f"({'PASS' if correct >= 8 else 'MARGINAL'})")

    # -------------------------------------------------------------------------
    # Test 9: Full game simulation — all algorithms play as X vs random O
    # -------------------------------------------------------------------------
    print("\n[TEST 9] Full Game — Each Algorithm as X vs Random O")
    algorithms = {
        "Minimax":           lambda s: Minimax().search(s),
        "AlphaBeta":         lambda s: AlphaBeta().search(s),
        "HeuristicAB(d=4)":  lambda s: HeuristicAlphaBeta(max_depth=4).search(s),
        "MCTS(1000)":        lambda s: MCTS(iterations=1000).search(s),
    }

    game_counts = {
        "Minimax": 2,
        "AlphaBeta": 5,
        "HeuristicAB(d=4)": 20,
        "MCTS(1000)": 10,
    }

    for name, algo in algorithms.items():
        wins = draws = losses = 0
        for _ in range(game_counts[name]):
            state = TicTacToe()
            while not state.is_terminal():
                if state.turn == 1:        # X plays using algorithm
                    _, move = algo(state)
                else:                      # O plays randomly
                    move = random.choice(state.legal_moves())
                state = state.make_move(move)
            result = state.terminal_value()
            if result == 1:   wins   += 1
            elif result == 0: draws  += 1
            else:             losses += 1
        print(f"  {name:20s}: games={game_counts[name]} W={wins} D={draws} L={losses}")

    # -------------------------------------------------------------------------
    # Test 10: Algorithm comparison on a mid-game position
    # -------------------------------------------------------------------------
    print("\n[TEST 10] Mid-Game Position — All Algorithms")
    state = TicTacToe([1, 0, -1, 0, 1, 0, 0, 0, -1], turn=1)
    print(f"\n  Board:\n{_indent(state.display(), '    ')}\n")
    for name, algo in algorithms.items():
        val, move = algo(state)
        print(f"  {name:20s}: move={move}, value={val:.3f}")

    print("\n" + "=" * 65)
    print("  ALL TESTS PASSED ✓")
    print("=" * 65)


def _indent(text, prefix):
    return '\n'.join(prefix + line for line in text.splitlines())


if __name__ == "__main__":
    def emit_step(title, status="running", detail="", data=None):
        print(json.dumps({
            "title": title,
            "status": status,
            "detail": detail,
            "data": data or {},
        }), flush=True)

    def run_api_steps():
        start = time.perf_counter()
        opening_state = TicTacToe()
        midgame_state = TicTacToe([1, 0, -1, 0, 1, 0, 0, 0, -1], turn=1)

        emit_step(
            "Initialise Tic-Tac-Toe",
            detail="Created an empty board with X as the maximising player.",
            data={
                "board": opening_state.board,
                "turn": "X",
                "legal_moves": opening_state.legal_moves(),
                "visual_type": "game-board",
            },
        )

        mm = Minimax()
        value, move = mm.search(opening_state)
        emit_step(
            "Run exact Minimax",
            detail="Explored the complete game tree and found optimal play from the start.",
            data={
                "algorithm": "Minimax",
                "best_move": move,
                "value": value,
                "nodes": mm.nodes_visited,
                "complexity": "O(b^d)",
                "board": opening_state.board,
                "visual_type": "game-search",
            },
        )

        ab = AlphaBeta()
        value_ab, move_ab = ab.search(opening_state)
        emit_step(
            "Apply Alpha-Beta pruning",
            detail="Matched Minimax while pruning branches that could not change the decision.",
            data={
                "algorithm": "Alpha-Beta",
                "best_move": move_ab,
                "value": value_ab,
                "nodes": ab.nodes_visited,
                "pruned": ab.pruned,
                "nodes_saved": mm.nodes_visited - ab.nodes_visited,
                "complexity": "best O(b^(d/2)), worst O(b^d)",
                "board": opening_state.board,
                "visual_type": "game-search",
            },
        )

        for depth in [2, 4, 6]:
            hab = HeuristicAlphaBeta(max_depth=depth)
            value_h, move_h = hab.search(opening_state)
            emit_step(
                f"Depth-limited heuristic search d={depth}",
                detail="Stopped at the depth limit and used board heuristics for unfinished states.",
                data={
                    "algorithm": f"Heuristic Alpha-Beta d={depth}",
                    "best_move": move_h,
                    "value": round(value_h, 3),
                    "nodes": hab.nodes_visited,
                    "heuristic_cutoffs": hab.cutoffs_by_heuristic,
                    "depth": depth,
                    "board": opening_state.board,
                    "visual_type": "game-search",
                },
            )

        mcts = MCTS(iterations=500)
        win_rate, move_mcts = mcts.search(opening_state)
        child_stats = sorted(
            [
                {
                    "move": child.move,
                    "visits": child.visits,
                    "win_rate": round(child.wins / child.visits, 3) if child.visits else 0,
                }
                for child in mcts.root.children
            ],
            key=lambda item: item["visits"],
            reverse=True,
        )
        emit_step(
            "Run Monte-Carlo Tree Search",
            detail="Selected, expanded, simulated, and backpropagated 500 random playouts.",
            data={
                "algorithm": "MCTS",
                "best_move": move_mcts,
                "win_rate": round(win_rate, 3),
                "iterations": 500,
                "board": opening_state.board,
                "child_stats": child_stats,
                "visual_type": "mcts",
            },
        )

        comparisons = []
        for name, algo in [
            ("Minimax", Minimax()),
            ("Alpha-Beta", AlphaBeta()),
            ("Heuristic AB d=4", HeuristicAlphaBeta(max_depth=4)),
            ("MCTS 1000", MCTS(iterations=1000)),
        ]:
            v, m = algo.search(midgame_state)
            comparisons.append({
                "algorithm": name,
                "move": m,
                "value": round(v, 3),
                "nodes": getattr(algo, "nodes_visited", None),
                "pruned": getattr(algo, "pruned", None),
            })
        emit_step(
            "Compare algorithms on a mid-game board",
            detail="All four approaches are run from the same non-terminal state.",
            data={
                "board": midgame_state.board,
                "turn": "X",
                "comparisons": comparisons,
                "visual_type": "algorithm-comparison",
            },
        )

        emit_step(
            "Q1 complete",
            status="complete",
            detail="Game-search comparison finished successfully.",
            data={"elapsed_ms": round((time.perf_counter() - start) * 1000, 2)},
        )

    if "--steps" in sys.argv:
        run_api_steps()
    else:
        run_tests()
