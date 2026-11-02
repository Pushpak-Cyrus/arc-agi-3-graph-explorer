"""
Graph Memory for CYRUS
Maintains a directed graph of explored states and state-action transitions.

Nodes = unique states (identified by hash)
Edges = actions and their outcomes
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Hashable
from collections import defaultdict, deque
import json


class StateNode:
    """Represents a single state in the graph."""

    def __init__(self, state_hash: str):
        """
        Args:
            state_hash: unique identifier for this state
        """
        self.hash = state_hash
        self.actions: Dict[Hashable, ActionEdge] = {}  # action signature -> ActionEdge
        self.visit_count = 0
        self.first_visited_step = None
        self.last_visited_step = None
        self.is_goal = False  # set to True if this state completed a level

    def add_action(self, action_sig: Hashable, priority_tier: int):
        """Register an action signature for this state.

        Action signatures may be simple IDs such as ``1`` or structured
        tuples such as ``(6, x, y)`` for coordinate-targeted actions.
        """
        if action_sig not in self.actions:
            self.actions[action_sig] = ActionEdge(action_sig, priority_tier)

    def get_untested_actions(self) -> List:
        """Return list of actions that haven't been tested from this state."""
        return [a for a in self.actions.values() if not a.tested]

    def get_actions_by_tier(self, tier: int) -> List:
        """Return actions at or above a given priority tier."""
        return [a for a in self.actions.values() if a.priority_tier <= tier]


class ActionEdge:
    """Represents an action and its outcome."""

    def __init__(self, action_sig, priority_tier: int):
        self.action_sig = action_sig
        self.priority_tier = priority_tier
        self.tested = False
        self.result_hash = None
        self.caused_reset = False
        self.high_value = False
        self.times_tested = 0
        self.replay_count = 0
        self.success_count = 0


class GraphMemory:
    """Maintains the state-action graph."""

    def __init__(self):
        self.nodes = {}  # hash -> StateNode
        self.step_counter = 0
        self.visited_path = []  # list of (state_hash, action signature) tuples in current episode
        self.visited_levels = []  # level id parallel to visited_path

        # Global, state-independent memory of how each action SIGNATURE has
        # performed across the whole run so far, keyed by action_sig.
        #
        # Why this exists: per-node (state_hash, action_sig) tracking is
        # only useful if the same state is revisited. In games where the
        # frame changes on almost every step (states-per-step ratio near
        # 1.0), a node is effectively visited once and never again, so a
        # high-value tag attached only to that node's edge is immediately
        # unreachable. This dict lets CYRUS remember "clicking around here
        # has caused progress before" regardless of which exact state it's
        # currently in.
        self.global_action_stats: Dict[Hashable, Dict[str, int]] = {}
        # Outcome history is scoped to a level because coordinate targets and
        # their meaning can change completely after a level transition.
        self.level_action_outcomes: Dict[int, Dict[Hashable, Dict[str, int]]] = {}
        self.action_trial_states: Dict[Hashable, Set[str]] = defaultdict(set)

    def add_state(self, state_hash: str) -> StateNode:
        """Add a state node if it doesn't exist. Returns the node."""
        if state_hash not in self.nodes:
            self.nodes[state_hash] = StateNode(state_hash)
        return self.nodes[state_hash]

    def record_transition(
            self,
            from_hash: str,
            action_sig: Hashable,
            to_hash: str,
            caused_reset: bool = False,
            level_id: int = 0,
            outcome_class: Optional[str] = None,
            replayed: bool = False,
    ):
        """
        Record a state-action-state transition.

        Args:
            from_hash: starting state
            action_sig: action signature taken, including coordinate targets
            to_hash: resulting state
            caused_reset: whether this action caused a level reset
        """
        from_node = self.add_state(from_hash)
        to_node = self.add_state(to_hash)

        # Mark action as tested
        edge = from_node.actions.get(action_sig)
        if edge:
            edge.tested = True
            edge.result_hash = to_hash
            edge.caused_reset = caused_reset
            if replayed:
                edge.replay_count += 1
            else:
                edge.times_tested += 1
                self.action_trial_states[action_sig].add(from_hash)

        # Record in path
        self.visited_path.append((from_hash, action_sig))
        self.visited_levels.append(int(level_id))
        self.step_counter += 1
        from_node.visit_count += 1
        from_node.last_visited_step = self.step_counter

        # Global, state-independent record of this signature being tried.
        global_stats = self.global_action_stats.setdefault(
            action_sig, {'times_tested': 0, 'success_count': 0,
                         'replay_count': 0}
        )
        if replayed:
            global_stats['replay_count'] = global_stats.get('replay_count', 0) + 1
        else:
            global_stats['times_tested'] += 1

        if not replayed:
            level_stats = self.level_action_outcomes.setdefault(
                int(level_id), {}
            ).setdefault(action_sig, {
                'attempts': 0,
                'confirmed_progress': 0,
                'visual_observation': 0,
                'no_confirmed_progress': 0,
                'negative_evidence': 0,
            })
            level_stats['attempts'] += 1
            if outcome_class in level_stats:
                level_stats[outcome_class] += 1

    def tag_high_value_edge(
            self,
            from_hash: str,
            action_sig: Hashable,
            globally_reusable: bool = True,
    ) -> None:
        """
        Mark an edge as high-value because it caused meaningful progress.
        This can be called for level advancement or other progress signals.

        By default, also records the signature in global_action_stats, so
        this discovery survives even if its exact (state, action) pair is
        never revisited again. Set ``globally_reusable=False`` for outcomes
        whose coordinate context changes, such as an action that advances
        to a new level.
        """
        node = self.nodes.get(from_hash)
        if node and action_sig in node.actions:
            node.actions[action_sig].high_value = True
            node.actions[action_sig].success_count += 1

        if globally_reusable:
            global_stats = self.global_action_stats.setdefault(
                action_sig, {'times_tested': 0, 'success_count': 0}
            )
            global_stats['success_count'] += 1

    @staticmethod
    def _action_sig_matches(
            a: Hashable,
            b: Hashable,
            coordinate_radius: int = 5,
    ) -> bool:
        """
        True if two action signatures should be treated as "the same
        experiment" - exact match for simple actions, or coordinates
        within ``coordinate_radius`` pixels for ACTION6-style targets
        (blob centroids drift slightly between near-identical states).
        """
        if a == b:
            return True

        if (
                isinstance(a, tuple) and isinstance(b, tuple)
                and len(a) >= 3 and len(b) >= 3
                and a[0] == b[0]
                and a[1] is not None and a[2] is not None
                and b[1] is not None and b[2] is not None
        ):
            return (
                abs(a[1] - b[1]) <= coordinate_radius
                and abs(a[2] - b[2]) <= coordinate_radius
            )

        return False

    def global_success_count(
            self,
            action_sig: Hashable,
            coordinate_radius: int = 5,
            level_id: Optional[int] = None,
    ) -> int:
        """
        Total success_count recorded for this action signature across
        ALL states seen so far, fuzzy-matching nearby click coordinates.
        """
        if level_id is not None:
            level_stats = self.level_action_outcomes.get(int(level_id), {})
            return sum(
                stats.get('confirmed_progress', 0)
                for sig, stats in level_stats.items()
                if self._action_sig_matches(action_sig, sig, coordinate_radius)
            )

        total = 0
        for sig, stats in self.global_action_stats.items():
            if self._action_sig_matches(action_sig, sig, coordinate_radius):
                total += stats['success_count']
        return total

    def is_globally_high_value(
            self,
            action_sig: Hashable,
            coordinate_radius: int = 5,
            level_id: Optional[int] = None,
    ) -> bool:
        """True if this action signature (or a nearby coordinate variant)
        has ever caused a progress signal anywhere in this run."""
        return self.global_success_count(
            action_sig,
            coordinate_radius,
            level_id=level_id,
        ) > 0

    def action_outcome_counts(
            self,
            action_sig: Hashable,
            level_id: int,
            coordinate_radius: int = 5,
    ) -> Dict[str, int]:
        """Return fuzzy-matched experiment outcomes from one level only."""
        totals = {
            'attempts': 0,
            'confirmed_progress': 0,
            'visual_observation': 0,
            'no_confirmed_progress': 0,
            'negative_evidence': 0,
        }
        for previous_sig, counts in self.level_action_outcomes.get(
                int(level_id), {}).items():
            if self._action_sig_matches(
                    action_sig, previous_sig, coordinate_radius):
                for key in totals:
                    totals[key] += counts.get(key, 0)
        return totals

    def recent_action_count(
            self,
            action_sig: Hashable,
            window_size: int = 32,
            coordinate_radius: int = 5,
            level_id: Optional[int] = None,
    ) -> int:
        """Count recent repeats of an action, grouping nearby click targets.

        ACTION6 blob centroids can drift slightly between frames. Treat
        coordinates within ``coordinate_radius`` pixels on both axes as the
        same recent target so a changing state hash does not make CYRUS click
        the same object repeatedly. Other action signatures match exactly.
        """
        if window_size <= 0:
            recent = []
        else:
            start = max(0, len(self.visited_path) - window_size)
            recent = [
                (idx, self.visited_path[idx][1])
                for idx in range(start, len(self.visited_path))
                if level_id is None or self.visited_levels[idx] == int(level_id)
            ]

        if (
                isinstance(action_sig, tuple)
                and len(action_sig) >= 3
                and action_sig[0] == 6
                and action_sig[1] is not None
                and action_sig[2] is not None
        ):
            action_id, x, y = action_sig[:3]
            return sum(
                1
                for _, previous_sig in recent
                if (
                    isinstance(previous_sig, tuple)
                    and len(previous_sig) >= 3
                    and previous_sig[0] == action_id
                    and previous_sig[1] is not None
                    and previous_sig[2] is not None
                    and abs(previous_sig[1] - x) <= coordinate_radius
                    and abs(previous_sig[2] - y) <= coordinate_radius
                )
            )

        return sum(1 for _, previous_sig in recent if previous_sig == action_sig)

    def get_or_create_action_from_blobs(self, state_hash: str, blobs: List) -> None:
        """
        Register all "blob-based" actions for a state.
        In click-based games, each blob is a potential clickable target.

        Args:
            state_hash: the state
            blobs: list of blob dicts from frame processor (each has priority_tier)
        """
        node = self.add_state(state_hash)

        # Preserve compatibility with the older blob-index representation,
        # while allowing callers to provide the richer coordinate-aware
        # signature used by CYRUS's current action selection.
        for blob_idx, blob in enumerate(blobs):
            priority_tier = blob.get('priority_tier', 3)
            action_sig = blob.get('action_sig')

            if action_sig is None:
                action_id = blob.get('action_id')
                if action_id is not None:
                    x = blob.get('x')
                    y = blob.get('y')
                    if action_id == 6 and x is not None and y is not None:
                        action_sig = (action_id, x, y)
                    else:
                        action_sig = (action_id, None, None)
                else:
                    # Legacy fallback. This remains only for old callers.
                    action_sig = f"blob_{blob_idx}"

            node.add_action(action_sig, priority_tier)

    def get_frontier(self, state_hash: str) -> Optional[Tuple[str, List]]:
        """
        Find the nearest state with untested actions, starting from the given state.
        Uses BFS to find the shortest path to a frontier.

        Returns:
            (frontier_state_hash, path_of_state_hashes) or None if no frontier exists
        """
        if state_hash not in self.nodes:
            return None

        current_node = self.nodes[state_hash]
        if current_node.get_untested_actions():
            return (state_hash, [state_hash])

        # BFS to find nearest frontier
        queue = deque([(state_hash, [state_hash])])
        visited_in_search = {state_hash}

        while queue:
            curr_hash, path = queue.popleft()
            curr_node = self.nodes[curr_hash]

            # Check all actions from this state
            for action in curr_node.actions.values():
                if action.result_hash and action.result_hash not in visited_in_search:
                    next_node = self.nodes.get(action.result_hash)
                    if next_node and next_node.get_untested_actions():
                        # Found a frontier!
                        return (action.result_hash, path + [action.result_hash])

                    visited_in_search.add(action.result_hash)
                    queue.append((action.result_hash, path + [action.result_hash]))

        return None

    def get_shallowest_frontier(
            self, root_hash: str, max_depth: Optional[int] = None
    ) -> Optional[Tuple[str, List[str]]]:
        """Return the shortest observed path from ``root_hash`` to an
        unexplored state, using BFS over recorded transitions.

        This is used after an environment RESET to resume the shallowest
        known frontier. Only observed edges are traversed; unknown transitions
        are never assumed.
        """
        if root_hash not in self.nodes:
            return None
        queue = deque([root_hash])
        parent = {root_hash: None}
        depth = {root_hash: 0}
        while queue:
            current = queue.popleft()
            node = self.nodes[current]
            if (node.get_untested_actions()
                    and (max_depth is None or depth[current] < max_depth)):
                path = []
                cursor = current
                while cursor is not None:
                    path.append(cursor)
                    cursor = parent[cursor]
                path.reverse()
                return current, path
            for edge in node.actions.values():
                target = edge.result_hash
                if (target and target in self.nodes and target not in parent
                        and (max_depth is None or depth[current] + 1 < max_depth)):
                    parent[target] = current
                    depth[target] = depth[current] + 1
                    queue.append(target)
        return None

    def get_best_frontier(self, root_hash: str) -> Optional[Tuple[str, List[str]]]:
        """Choose a reachable frontier by expansion value per replay cost.

        Value is the number of currently untested actions. The deterministic
        score is ``value / (path_cost + 1)``; ties prefer more value, then a
        cheaper replay. This keeps the graph search cost-sensitive without
        relying on game-specific information.
        """
        if root_hash not in self.nodes:
            return None
        queue = deque([root_hash])
        parent = {root_hash: None}
        candidates = []
        while queue:
            current = queue.popleft()
            node = self.nodes[current]
            value = len(node.get_untested_actions())
            if value:
                path = []
                cursor = current
                while cursor is not None:
                    path.append(cursor)
                    cursor = parent[cursor]
                path.reverse()
                cost = len(path) - 1
                candidates.append((value / (cost + 1), value, -cost,
                                   current, path))
            for edge in node.actions.values():
                target = edge.result_hash
                if target and target in self.nodes and target not in parent:
                    parent[target] = current
                    queue.append(target)
        if not candidates:
            return None
        best = max(candidates, key=lambda row: row[:3])
        return best[3], best[4]

    def get_shortest_path(self, from_hash: str, to_hash: str) -> Optional[List[str]]:
        """
        Get shortest path from one state to another using BFS.

        Returns:
            list of state hashes, or None if no path exists
        """
        if from_hash == to_hash:
            return [from_hash]

        queue = deque([(from_hash, [from_hash])])
        visited = {from_hash}

        while queue:
            curr_hash, path = queue.popleft()
            curr_node = self.nodes.get(curr_hash)

            if not curr_node:
                continue

            for action in curr_node.actions.values():
                if action.result_hash and action.result_hash not in visited:
                    if action.result_hash == to_hash:
                        return path + [to_hash]

                    visited.add(action.result_hash)
                    queue.append((action.result_hash, path + [action.result_hash]))

        return None

    def distinct_state_trials(self, action_sig: Hashable,
                              level_id: Optional[int] = None) -> int:
        """Count distinct source states where a signature was newly tried."""
        return len(self.action_trial_states.get(action_sig, ()))

    def get_stats(self) -> Dict:
        """Return summary statistics about the graph."""
        total_actions = sum(len(node.actions) for node in self.nodes.values())
        tested_actions = sum(
            sum(1 for a in node.actions.values() if a.tested)
            for node in self.nodes.values()
        )
        high_value_actions = sum(
            sum(1 for a in node.actions.values() if a.high_value)
            for node in self.nodes.values()
        )

        return {
            'num_states': len(self.nodes),
            'total_actions': total_actions,
            'tested_actions': tested_actions,
            'untested_actions': total_actions - tested_actions,
            'high_value_actions': high_value_actions,
            'steps_taken': self.step_counter,
        }

    def is_loop_detected(self, state_hash: str, window_size: int = 10) -> bool:
        """
        Detect if we're stuck in a loop: revisiting the same state repeatedly.

        Args:
            state_hash: current state
            window_size: how many recent steps to check

        Returns:
            True if this state has been visited multiple times recently
        """
        if len(self.visited_path) < window_size:
            return False

        recent_states = [s for s, _ in self.visited_path[-window_size:]]
        return recent_states.count(state_hash) >= 3  # appeared 3+ times in last N steps

    def is_stalled(self, steps_threshold: int = 50) -> bool:
        """
        Detect if we haven't discovered a new state in a while.

        Args:
            steps_threshold: how many steps without new state discovery

        Returns:
            True if we haven't seen a new state recently
        """
        if len(self.visited_path) < steps_threshold:
            return False

        recent_states = set(s for s, _ in self.visited_path[-steps_threshold:])
        return len(recent_states) <= 2  # only 1-2 unique states in last N steps

    @staticmethod
    def _serialize_action_signature(action_sig: Hashable) -> Dict[str, Any]:
        """Convert an action signature into a JSON-safe representation."""
        if isinstance(action_sig, tuple):
            return {
                'type': 'tuple',
                'value': list(action_sig),
            }
        return {
            'type': 'scalar',
            'value': action_sig,
        }

    def to_dict(self) -> Dict:
        """Serialize the graph to a JSON-safe dict for saving/debugging.

        Structured signatures such as ``(6, x, y)`` cannot be JSON object
        keys, so actions are stored as a list of edge records instead of
        using the action signature directly as a dictionary key.
        """
        nodes_dict = {}
        for hash_val, node in self.nodes.items():
            actions = []
            for action_sig, edge in node.actions.items():
                actions.append({
                    'action_sig': self._serialize_action_signature(action_sig),
                    'tested': edge.tested,
                    'result_hash': edge.result_hash,
                    'caused_reset': edge.caused_reset,
                    'high_value': edge.high_value,
                    'priority_tier': edge.priority_tier,
                    'times_tested': edge.times_tested,
                    'replay_count': edge.replay_count,
                    'success_count': edge.success_count,
                })

            nodes_dict[hash_val] = {
                'visit_count': node.visit_count,
                'is_goal': node.is_goal,
                'actions': actions,
            }

        return {
            'nodes': nodes_dict,
            'step_counter': self.step_counter,
            'stats': self.get_stats(),
        }

# Example usage
if __name__ == "__main__":
    graph = GraphMemory()

    # Simulate some transitions
    graph.add_state("state_1")
    graph.add_state("state_2")
    graph.record_transition("state_1", "action_0", "state_2")

    # Tag an edge as high-value
    graph.tag_high_value_edge("state_1", "action_0")

    # Print stats
    stats = graph.get_stats()
    print(f"Graph stats: {stats}")
