import unittest
import numpy as np

from active_drone.inference import StructuredBlanketLearner
from active_drone.structure import PARTITIONS, NODES, architectural_gap, edge_template
from active_drone.simulation import make_system, run_mission
from active_drone.world import DesignAction
from active_drone.collective import CollaborativeTeam, CollectiveWorld, make_collective_system, run_collective_mission


class DroneV5Tests(unittest.TestCase):
    def test_exact_structured_state_space(self):
        self.assertEqual(PARTITIONS.shape, (2520, NODES))
        self.assertTrue(np.all(np.stack([(PARTITIONS == r).sum(1) for r in range(4)]) == 2))

    def test_joint_posterior_normalizes(self):
        learner = StructuredBlanketLearner()
        self.assertAlmostEqual(float(learner.posterior.sum()), 1.0)
        np.testing.assert_allclose(learner.marginals.sum(1), 1.0)

    def test_clean_template_has_zero_path_gap(self):
        roles = PARTITIONS[0]
        self.assertAlmostEqual(architectural_gap(edge_template(roles), roles), 0.0)

    def test_constitutive_shield_changes_world(self):
        world, _ = make_system(3, "constitutive", "construct")
        pair = next(iter(world.leaks))
        action = DesignAction("shield", min(pair), max(pair))
        before = world.matrix.copy()
        out = world.step(0.0, action, constitutive=True)
        self.assertTrue(out["design_changed_causal_matrix"])
        self.assertGreater(np.max(abs(world.matrix - before)), 0)

    def test_diagnostic_action_does_not_change_world(self):
        world, _ = make_system(3, "diagnostic_only", "construct")
        pair = next(iter(world.leaks))
        before = world.matrix.copy()
        out = world.step(0.0, DesignAction("shield", min(pair), max(pair)), constitutive=False)
        self.assertFalse(out["design_changed_causal_matrix"])
        np.testing.assert_allclose(world.matrix, before)

    def test_mission_terminates(self):
        world, drone = make_system(0, "constitutive", "construct")
        result = run_mission(world, drone)
        self.assertLessEqual(len(result.records), world.horizon)

    def test_collective_posterior_normalizes(self):
        _, team = make_collective_system(4, "collective_constitutive", "assemble")
        self.assertAlmostEqual(float(team.pooled_posterior.sum()), 1.0)
        np.testing.assert_allclose(team.pooled_marginals.sum(1), 1.0)

    def test_unilateral_collective_action_cannot_change_boundary(self):
        world, _ = make_collective_system(5, "collective_constitutive", "assemble")
        pair = next(iter(world.leaks)); action = DesignAction("shield", *pair)
        before = world.matrix.copy()
        out = world.step(np.zeros(2), (action, DesignAction()), constitutive=True)
        self.assertFalse(out["agreement"])
        self.assertFalse(out["design_changed_causal_matrix"])
        np.testing.assert_allclose(world.matrix, before)

    def test_bilateral_collective_action_changes_boundary(self):
        world, _ = make_collective_system(5, "collective_constitutive", "assemble")
        pair = next(iter(world.leaks)); action = DesignAction("shield", *pair)
        out = world.step(np.zeros(2), (action, action), constitutive=True)
        self.assertTrue(out["agreement"])
        self.assertTrue(out["design_changed_causal_matrix"])

    def test_collective_mission_terminates(self):
        world, team = make_collective_system(0, "collective_constitutive", "reconfigure")
        result = run_collective_mission(world, team)
        self.assertLessEqual(len(result.records), world.horizon)


if __name__ == "__main__":
    unittest.main()
