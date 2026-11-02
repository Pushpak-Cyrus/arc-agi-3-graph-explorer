"""Tests for action exposure and bounded component/edge click probes."""

import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))
sys.path.insert(0, str(ROOT / "Cyrus"))

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


if __name__ == "__main__":
    unittest.main()
