"""Focused tests for verified multi-step movement planning."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from Cyrus.system2.action_effect_learner import ActionEffectLearner
from Cyrus.system2.goal_inference import GoalHypothesis
from Cyrus.system2.system2 import System2


def _pixel(row, col, color):
    return {
        "cells": frozenset({(row, col)}),
        "bbox": (row, col, row, col),
        "center": (float(row), float(col)),
        "width": 1,
        "height": 1,
        "area": 1,
        "color_count": 1,
        "primary_value": color,
    }


def _state(mover_col=1):
    grid = np.zeros((8, 8), dtype=np.int16)
    grid[3, mover_col] = 2
    grid[3, 5] = 2
    mover, target = _pixel(3, mover_col, 2), _pixel(3, 5, 2)
    return {
        "grid": grid,
        "masked_grid": grid.copy(),
        "blobs": [
            {"coords": [(3, mover_col)], "primary_value": 2},
            {"coords": [(3, 5)], "primary_value": 2},
        ],
        "targets": [],
        "objects": [mover, target],
        "relations": [],
    }


class SequencePlannerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.system = System2()
        self.system.goals.memory_path = Path(self.temp_dir.name) / "goals.json"

    def _install_right_move(self, state):
        learner = self.system.action_effects
        context = learner.context(None, None, state, sig=(1, None, None))
        shape = (1, 1, ((0, 0),))
        object_context = learner.object_context(state, 0)
        signature = (shape, object_context, (0, 1))
        key = (1, context)
        learner.trials[key] = 2
        learner.changed_trials[key] = 2
        learner.effect_trials[key][signature] = 2
        pred_key = (1, context, signature)
        learner.prediction_trials[pred_key] = 2
        learner.prediction_correct[pred_key] = 2
        return context, pred_key

    def test_bfs_finds_two_step_plan_to_supported_relation_goal(self):
        state = _state()
        self._install_right_move(state)
        object_class = (1, 1, 4, 1)
        name = "adjacent_pixels"
        self.system.goals.goals[name] = GoalHypothesis(
            name=name,
            kind="relation",
            colours=(),
            relation_key=("adjacent", object_class, object_class),
            true_obs=10,
            true_progress=2,
        )
        self.system.goals.total_obs = 20
        self.system.goals.total_progress = 2

        plan = self.system._plan_sequence(state, [(1, None, None)])

        self.assertIsNotNone(plan)
        self.assertEqual(plan["goal"], name)
        self.assertEqual(plan["actions"], [(1, None, None), (1, None, None)])

    def test_prediction_mismatch_retires_effect_from_verified_planning(self):
        state = _state()
        learner = self.system.action_effects
        context, pred_key = self._install_right_move(state)
        self.assertEqual(len(learner.verified_effects(1, context)), 1)

        no_move = _state()
        no_move["grid"][0, 0] = 7
        no_move["masked_grid"][0, 0] = 7
        no_move["blobs"].append({"coords": [(0, 0)], "primary_value": 7})
        learner.observe(
            (1, None, None), state, no_move, learn_motion=False
        )

        self.assertEqual(learner.verified_effects(1, context), [])
        self.assertIn(pred_key, learner.invalidated_effects)

    def test_rigid_group_passive_drift_and_blockers_are_exposed(self):
        learner = ActionEffectLearner(min_trials=2)
        context = ("visible context",)
        shape_a = (1, 1, ((0, 0),))
        shape_b = (1, 2, ((0, 0), (0, 1)))
        effect_a = (shape_a, "left object", (0, 1))
        effect_b = (shape_b, "right object", (0, 1))
        for effect in (effect_a, effect_b):
            key = (1, context)
            learner.trials[key] = 2
            learner.effect_trials[key][effect] = 2
            pred_key = (1, context, effect)
            learner.prediction_trials[pred_key] = 2
            learner.prediction_correct[pred_key] = 2
        group = tuple(sorted((effect_a, effect_b), key=repr))
        learner.rigid_group_trials[(1, context, group)] = 2
        learner.effect_action_trials[effect_a][1] = 2
        learner.effect_action_trials[effect_a][2] = 2
        learner.effect_action_trials[effect_a][3] = 2
        learner.blocked_color_trials[(1, context, shape_a, (0, 1), 9)] = 2

        self.assertEqual(len(learner.rigid_groups(1, context)), 1)
        self.assertTrue(learner.is_passive_effect(
            2, {"shape": shape_a, "object_context": "left object",
                "vector": (0, 1)}
        ))
        self.assertEqual(
            learner.blocked_colors(1, context, shape_a, (0, 1)), {9: 2}
        )


if __name__ == "__main__":
    unittest.main()
