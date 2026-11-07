"""Tests for action exposure and bounded component/edge click probes."""

import sys
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))
sys.path.insert(0, str(ROOT / "Cyrus"))

from agents.agent import GameAction
from cyrus_agent import CyrusAgent
from graph_memory import GraphMemory


def _agent_for_policy():
    agent = CyrusAgent.__new__(CyrusAgent)
    agent.movement_probe_trials = Counter()
    agent.movement_probe_progress = set()
    agent.click_class_trials = Counter()
    agent.click_class_progress = set()
    agent.empty_click_trials = 0
    agent.empty_click_progress = False
    agent.click_class_changed = set()
    agent.click_class_noop_states = {}
    agent.graph = GraphMemory()
    return agent


def _state():
    grid = np.zeros((64, 64), dtype=np.int16)
    grid[10, 10] = 9
    obj = {
        "cells": frozenset({(10, 10)}),
        "bbox": (10, 10, 10, 10),
        "center": (10.0, 10.0),
        "width": 1,
        "height": 1,
        "area": 1,
        "color_count": 1,
    }
    return {
        "hash": "synthetic-state",
        "grid": grid,
        "masked_grid": grid,
        "blobs": [{"coords": [(10, 10)], "priority_tier": 0}],
        "objects": [obj],
    }


class ActionProbePolicyTests(unittest.TestCase):
    def test_system1_learned_ranking_has_an_independent_gate(self):
        agent = _agent_for_policy()

        class LearnedSignals:
            def value(self, _sig, _state):
                return 0.4

        class GoalSignals:
            def relevance(self, _state, _sig):
                return 0.25

        class Components:
            interactions = LearnedSignals()
            goals = GoalSignals()

        agent.system2 = Components()
        agent.learned_ranking_enabled = False
        self.assertEqual(
            agent._learned_ranking_key((1, None, None), {}), (0.0, 0.0)
        )

        agent.learned_ranking_enabled = True
        self.assertEqual(
            agent._learned_ranking_key((1, None, None), {}), (-0.4, -0.25)
        )

    def test_edges_are_deferred_until_component_representatives_are_tried(self):
        agent = _agent_for_policy()
        actions = [1, 2, 3, 4, 6]
        state = _state()
        node = agent.graph.add_state(state["hash"])
        candidates = agent._get_action_signatures(actions, state)
        signatures = {sig for sig, _tier in candidates}
        self.assertEqual({sig[0] for sig in signatures}, {1, 2, 3, 4, 6})
        self.assertNotIn((6, 31, 63), signatures)

        node.add_action((6, 10, 10), 0)
        agent.graph.record_transition(state["hash"], (6, 10, 10), "next")
        candidates = agent._get_action_signatures(actions, state)
        self.assertIn((6, 31, 63), {sig for sig, _tier in candidates})

    def test_prior_click_probe_does_not_hide_a_candidate(self):
        agent = _agent_for_policy()
        click = (6, 10, 10)
        class_key = (1, 1, 4, 1)
        agent.click_class_trials[class_key] = 1

        candidates = agent._get_action_signatures([6], _state())
        self.assertIn(click, {sig for sig, _tier in candidates})

    def test_iterative_widening_keeps_clicks_back_until_simple_actions_are_tried(self):
        agent = _agent_for_policy()
        agent.frontier_mode_enabled = True
        agent.search_order_enabled = True
        state = _state()
        node = agent.graph.add_state(state["hash"])
        node.add_action((1, None, None), 0)
        node.add_action((2, None, None), 0)

        candidates = agent._get_action_signatures([1, 2, 6], state)
        signatures = {sig for sig, _tier in candidates}
        self.assertNotIn((6, 10, 10), signatures)

        agent.graph.record_transition(state["hash"], (1, None, None), "s1")
        agent.graph.record_transition(state["hash"], (2, None, None), "s2")
        candidates = agent._get_action_signatures([1, 2, 6], state)
        self.assertIn((6, 10, 10), {sig for sig, _tier in candidates})

    def test_exhausted_root_selects_other_frontier_without_zero_length_attempt(self):
        agent = _agent_for_policy()
        agent.frontier_mode_enabled = True
        agent.frontier_depth_schedule = (6, 12, 24, 50)
        agent.frontier_depth_index = 0
        agent.frontier_depth_limit = 6
        agent.frontier_attempt_cap = 6
        agent.frontier_attempt_actions = 0
        agent.frontier_attempts = []
        agent.zero_length_frontier_attempts = 0
        agent._attempt_start_hash = None
        agent.frontier_root_hash = "root"
        agent.frontier_replay_target = None
        agent.last_action_signature = (0, None, None)
        agent.actions_since_reset = 0
        agent.actions_since_level_start = 0

        root = agent.graph.add_state("root")
        child = agent.graph.add_state("child")
        root.add_action((1, None, None), 0)
        agent.graph.record_transition("root", (1, None, None), "child")
        child.add_action((1, None, None), 0)

        agent._close_frontier_attempt("root", "state_fully_explored")
        frontier = agent._find_frontier_with_iterative_deepening("root")

        self.assertIsNotNone(frontier)
        self.assertEqual(agent.frontier_replay_target, "child")
        self.assertFalse(agent._frontier_reset_allowed())
        self.assertTrue(agent._replaying_to_frontier("root"))
        self.assertFalse(agent._replaying_to_frontier("child"))
        self.assertEqual(agent.frontier_attempts, [])
        self.assertEqual(agent.zero_length_frontier_attempts, 0)

    def test_exhausted_root_after_reset_does_not_create_zero_length_attempt(self):
        agent = _agent_for_policy()
        agent.frontier_mode_enabled = True
        agent.frontier_depth_schedule = (6, 12, 24, 50)
        agent.frontier_depth_index = 0
        agent.frontier_depth_limit = 6
        agent.frontier_attempt_cap = 6
        agent.frontier_attempt_actions = 0
        agent.frontier_attempts = []
        agent.zero_length_frontier_attempts = 0
        agent._attempt_start_hash = None
        agent.frontier_root_hash = "root"
        agent.frontier_replay_target = None
        agent.frontier_search_exhausted = False
        agent.last_action_signature = (0, None, None)
        agent.actions_since_reset = 0
        agent.actions_since_level_start = 0
        agent._just_reset_after_game_over = False
        agent.last_levels_completed = 0
        agent.last_score = None
        agent.first_level_advance_action = None
        agent.DEBUG_FRAME_DIFF = False
        agent.action_counter = 0
        agent.relational_planner_enabled = False
        agent.system2_takeover_enabled = False
        agent._post_reset_action_pending = False
        agent.movement_probe_trials = Counter()
        agent.budget = SimpleNamespace(record_step=Mock())
        agent.frame_processor = SimpleNamespace(
            process=lambda _frame: {"hash": "root"}
        )
        agent.goal_tracker = SimpleNamespace(summarize_state=lambda _state: {})
        agent.experiment_log = SimpleNamespace(begin=Mock())
        agent.system2 = SimpleNamespace(watch=Mock())
        agent._record_observation = lambda *_args, **_kwargs: (None, False)
        agent._get_action_signatures = lambda *_args: [
            ((1, None, None), 0)
        ]
        agent._select_action = lambda *_args: (1, None, None)
        agent._record_probe_attempt = Mock()

        root = agent.graph.add_state("root")
        root.add_action((1, None, None), 0)
        agent.graph.record_transition("root", (1, None, None), "root")
        frame = SimpleNamespace(
            state=SimpleNamespace(name="NOT_FINISHED"),
            levels_completed=0,
            frame=np.zeros((1, 1), dtype=np.int16),
            available_actions=[1],
        )

        action = agent.choose_action([], frame)

        self.assertIsNot(action, GameAction.RESET)
        self.assertTrue(agent.frontier_search_exhausted)
        self.assertEqual(agent.zero_length_frontier_attempts, 0)
        self.assertEqual(agent.frontier_attempts, [])

    def test_search_exhaustion_does_not_suppress_required_game_over_reset(self):
        agent = _agent_for_policy()
        agent.frontier_search_exhausted = True
        agent.budget = SimpleNamespace(should_stop_run=Mock(return_value=False))
        agent.action_counter = 0
        agent.MAX_ACTIONS = 100
        agent._post_reset_action_pending = False
        agent.consecutive_reset_stalls = 0
        agent.experiment_log = SimpleNamespace(pending=None)
        frame = SimpleNamespace(
            state=SimpleNamespace(name="GAME_OVER"),
            levels_completed=0,
        )

        self.assertFalse(agent.is_done([], frame))

    def test_unreachable_replay_target_stays_pending_and_blocks_reset(self):
        agent = _agent_for_policy()
        agent.search_order_enabled = True
        agent.frontier_replay_target = "unreachable"
        agent.frontier_replay_misses = 0
        agent.action_noop_states = {}
        agent.noop_deprioritize_after = 3
        agent.learned_ranking_enabled = False
        agent.learned_ranking_decisions = 0
        agent.learned_ranking_choice_changes = 0
        agent.last_action_signature = (1, None, None)
        agent.frontier_reset_due = True
        node = agent.graph.add_state("root")
        node.add_action((1, None, None), 0)

        action = agent._select_action(
            "root", node, _state(), SimpleNamespace(available_actions=[1])
        )

        self.assertEqual(action, (1, None, None))
        self.assertEqual(agent.frontier_replay_target, "unreachable")
        self.assertEqual(agent.frontier_replay_misses, 1)
        self.assertTrue(agent._replaying_to_frontier("root"))

    def test_unreachable_frontier_is_replaced_after_three_missed_replays(self):
        agent = _agent_for_policy()
        agent.search_order_enabled = True
        agent.frontier_mode_enabled = True
        agent.frontier_depth_schedule = (6, 12, 24, 50)
        agent.frontier_depth_index = 0
        agent.frontier_depth_limit = 6
        agent.frontier_attempt_cap = 6
        agent.frontier_root_hash = "root"
        agent.frontier_replay_target = "unreachable"
        agent.frontier_replay_misses = 0
        agent.action_noop_states = {}
        agent.noop_deprioritize_after = 3
        agent.learned_ranking_enabled = False
        agent.learned_ranking_decisions = 0
        agent.learned_ranking_choice_changes = 0
        agent.last_action_signature = (1, None, None)

        root = agent.graph.add_state("root")
        alternate = agent.graph.add_state("alternate")
        root.add_action((1, None, None), 0)
        agent.graph.record_transition("root", (1, None, None), "alternate")
        alternate.add_action((2, None, None), 0)
        state = _state()
        state["hash"] = "root"

        for _ in range(3):
            agent._select_action(
                "root", root, state, SimpleNamespace(available_actions=[1, 2])
            )

        self.assertEqual(agent.frontier_replay_target, "alternate")
        self.assertEqual(agent.frontier_replay_misses, 0)

    def test_unreachable_frontier_uses_previous_clear_behavior_when_flag_off(self):
        agent = _agent_for_policy()
        agent.search_order_enabled = False
        agent.frontier_replay_target = "unreachable"
        agent.frontier_replay_misses = 0
        agent.action_noop_states = {}
        agent.noop_deprioritize_after = 3
        agent.learned_ranking_enabled = False
        agent.learned_ranking_decisions = 0
        agent.learned_ranking_choice_changes = 0
        agent.last_action_signature = (1, None, None)
        node = agent.graph.add_state("root")
        node.add_action((1, None, None), 0)

        agent._select_action(
            "root", node, _state(), SimpleNamespace(available_actions=[1])
        )

        self.assertIsNone(agent.frontier_replay_target)
        self.assertEqual(agent.frontier_replay_misses, 0)


if __name__ == "__main__":
    unittest.main()
