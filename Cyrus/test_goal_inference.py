"""Regression tests for evidence-calibrated goal weighting."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from Cyrus.system2.goal_inference import GoalHypothesis, GoalInference


class GoalWeightCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.inference = GoalInference(
            memory_path=str(Path(self.temp_dir.name) / "goal-memory.json")
        )

    def test_ar25_step_178_untested_clear_goals_get_no_weight(self):
        """Reproduce ar25's pre-completion log with untested clear goals.

        At step 178 the run had 178 observations and no progress. cleared
        candidates were untested; contact candidates had many observations
        but no progress. None should get action-ranking weight.
        """
        inference = self.inference
        inference.total_obs = 178
        inference.total_progress = 0
        inference.goals = {
            "cleared(10)": GoalHypothesis("cleared(10)", "cleared", (10,)),
            "cleared(11)": GoalHypothesis("cleared(11)", "cleared", (11,)),
            "cleared(5)": GoalHypothesis("cleared(5)", "cleared", (5,)),
            "contact(10,5)": GoalHypothesis(
                "contact(10,5)", "contact", (10, 5), true_obs=84
            ),
        }

        self.assertEqual(inference.weights(), {})
        self.assertEqual(inference.top(k=12), [])

    def test_goal_needs_minimum_support_and_positive_lift(self):
        inference = self.inference
        inference.total_obs = 178
        inference.total_progress = 3
        inference.goals = {
            "too_rare": GoalHypothesis(
                "too_rare", "cleared", (1,), true_obs=1, true_progress=1
            ),
            "no_lift": GoalHypothesis(
                "no_lift", "contact", (1, 2), true_obs=40, true_progress=0
            ),
            "supported_lift": GoalHypothesis(
                "supported_lift", "relation", (), true_obs=20, true_progress=2
            ),
        }

        weights = inference.weights()
        self.assertEqual(set(weights), {"supported_lift"})
        self.assertEqual(weights["supported_lift"], 1.0)

    def test_ar25_trace_has_no_contrastive_contact_goal(self):
        # Logged goal candidates at steps 178-180 and the pre-action success
        # state at 181 have identical truth values for all 12 candidates.
        names = (
            "contact(0,5)", "contact(10,5)", "contact(11,4)",
            "contact(11,5)", "contact(4,11)", "contact(4,5)",
            "contact(5,0)", "contact(5,10)", "contact(5,11)",
            "contact(5,4)", "cleared(10)", "cleared(11)",
        )
        success = {name: name in {
            "contact(0,5)", "contact(10,5)", "contact(11,4)",
            "contact(11,5)", "contact(4,11)", "contact(5,0)",
            "contact(5,10)", "contact(5,11)",
        } for name in names}
        negatives = [dict(success) for _ in range(3)]

        self.assertEqual(
            GoalInference._minimal_contrastive_set(success, negatives), ()
        )

    def test_r11l_trace_does_not_credit_all_cooccurring_contacts(self):
        names = (
            "contact(0,15)", "contact(0,2)", "contact(1,15)",
            "contact(1,3)", "contact(15,0)", "contact(15,1)",
            "contact(15,3)", "contact(15,6)", "contact(2,0)",
            "contact(3,1)", "contact(3,15)", "contact(6,15)",
        )
        success = dict.fromkeys(names, True)
        # Candidate status snapshots at steps 27, 28, and 29. The final
        # negative is indistinguishable from the success among these logged
        # predicates, so no set can honestly identify a cause.
        n27 = dict.fromkeys(names, True)
        n27.update({"contact(1,3)": False, "contact(3,1)": False})
        n28 = dict.fromkeys(names, True)
        n28.update({
            "contact(1,15)": False, "contact(15,1)": False,
            "contact(1,3)": False, "contact(3,1)": False,
        })
        n29 = dict(success)

        self.assertEqual(
            GoalInference._minimal_contrastive_set(success, [n27, n28, n29]),
            (),
        )

    def test_minimal_separating_set_is_selected(self):
        success = {"a": True, "b": True, "c": True, "d": True}
        negatives = [
            {"a": False, "b": True, "c": True, "d": False},
            {"a": True, "b": False, "c": True, "d": False},
            {"a": True, "b": True, "c": False, "d": False},
        ]

        self.assertEqual(
            GoalInference._minimal_contrastive_set(success, negatives), ("d",)
        )

    def test_credited_goal_is_removed_after_unsuccessful_retest(self):
        inference = self.inference
        goal = GoalHypothesis("cleared(1)", "cleared", (1,))
        inference.goals = {goal.name: goal}
        prior = {"grid": np.array([[1, 0], [0, 0]])}
        success = {"grid": np.array([[0, 0], [0, 0]])}

        inference.observe(prior, progressed=False)
        inference.observe(success, progressed=True)
        self.assertTrue(goal.pending_retest)
        self.assertEqual(goal.contrastive_credits, 1)

        inference.observe(success, progressed=False)
        self.assertEqual(goal.retest_failures, 1)
        self.assertFalse(goal.pending_retest)
        self.assertEqual(inference.weights(), {})

    @staticmethod
    def _obj(cells):
        cells = frozenset(cells)
        rows = [row for row, _ in cells]
        cols = [col for _, col in cells]
        return {
            "cells": cells,
            "width": max(cols) - min(cols) + 1,
            "height": max(rows) - min(rows) + 1,
            "area": len(cells),
            "color_count": 1,
            "center": (
                sum(rows) / len(rows),
                sum(cols) / len(cols),
            ),
            "bbox": (min(rows), min(cols), max(rows), max(cols)),
        }

    def test_generic_marker_goals_are_inferred_from_motion_and_static_objects(self):
        inference = self.inference
        frames = []
        for col in (1, 2, 3):
            frames.append({
                "grid": np.zeros((64, 64), dtype=np.int16),
                "objects": [
                    self._obj({(1, col), (1, col + 1), (2, col), (2, col + 1)}),
                    self._obj({(20, 20)}),
                    self._obj({(20, 24)}),
                ],
                "relations": [],
            })
        for state in frames:
            inference.ensure_goals(state)
            inference.observe(state, progressed=False)

        kinds = {goal.kind for goal in inference.goals.values()}
        self.assertIn("cover_markers", kinds)
        self.assertIn("touch_markers", kinds)
        self.assertTrue(any(
            goal.transform == "mirror_x"
            for goal in inference.goals.values()
            if goal.kind == "cover_markers"
        ))

    def test_cover_and_mirror_footprints_hold_from_observed_cells(self):
        mover = self._obj({(1, 1), (1, 2), (2, 1), (2, 2)})
        state = {
            "grid": np.zeros((64, 64), dtype=np.int16),
            "objects": [mover],
        }
        identity = GoalHypothesis(
            "cover", "cover_markers", (), mover_class=(2, 2, 4, 1),
            marker_groups=(((1, 1),), ((2, 2),)),
        )
        mirrored = GoalHypothesis(
            "mirror", "cover_markers", (), mover_class=(2, 2, 4, 1),
            marker_groups=(((1, 61),), ((2, 62),)), transform="mirror_x",
        )

        self.assertTrue(identity.holds(state))
        self.assertTrue(mirrored.holds(state))


if __name__ == "__main__":
    unittest.main()

