import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from trajectory_guidance.mock_smolvla import TinySmolVLAPolicy, make_tiny_batch
from trajectory_guidance.trajectory_conditioning import (
    GeometryValidityGate,
    SmolVLATrajectoryWrapper,
    TrajectoryConditioningConfig,
)


class TrajectoryConditioningTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.config = TrajectoryConditioningConfig(
            num_waypoints=4, encoder_hidden_dim=12, gate_hidden_dim=8
        )

    def test_missing_trajectory_is_exact_base_fallback(self):
        base = TinySmolVLAPolicy()
        batch = make_tiny_batch(num_waypoints=4, include_trajectory=False)
        expected = base.predict_action_chunk(batch).clone()
        wrapper = SmolVLATrajectoryWrapper(base, self.config)
        actual = wrapper.predict_action_chunk(batch)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        self.assertIs(wrapper.config, base.config)

    def test_e2_all_invalid_trajectory_is_exact_base_fallback(self):
        config = TrajectoryConditioningConfig(
            num_waypoints=4, gate_mode="geometry", encoder_hidden_dim=12
        )
        base = TinySmolVLAPolicy()
        batch = make_tiny_batch(num_waypoints=4)
        batch["trajectory.local_waypoints_valid"].zero_()
        expected = base.predict_action_chunk(batch).clone()
        wrapper = SmolVLATrajectoryWrapper(base, config)
        actual = wrapper.predict_action_chunk(batch)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_valid_trajectory_changes_output_and_preserves_shape(self):
        base = TinySmolVLAPolicy()
        batch = make_tiny_batch(num_waypoints=4)
        expected = base.predict_action_chunk(batch).clone()
        wrapper = SmolVLATrajectoryWrapper(base, self.config)
        actual = wrapper.predict_action_chunk(batch)
        self.assertEqual(actual.shape, expected.shape)
        self.assertFalse(torch.equal(actual, expected))

    def test_adapter_parameters_receive_finite_gradients(self):
        wrapper = SmolVLATrajectoryWrapper(TinySmolVLAPolicy(), self.config)
        batch = make_tiny_batch(num_waypoints=4)
        loss, _ = wrapper(batch)
        loss.backward()
        gradients = [parameter.grad for parameter in wrapper.adapter_parameters()]
        self.assertTrue(gradients)
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        self.assertTrue(any(torch.count_nonzero(gradient) > 0 for gradient in gradients))

    def test_geometry_anomaly_lowers_gain(self):
        gate_config = TrajectoryConditioningConfig(
            num_waypoints=4, gate_mode="geometry", gate_hidden_dim=8
        )
        gate = GeometryValidityGate(gate_config)
        normal = make_tiny_batch(num_waypoints=4)
        anomalous = make_tiny_batch(num_waypoints=4, anomalous=True)
        normal_gain, _ = gate(
            normal["trajectory.local_waypoints_xyz_m"],
            normal["trajectory.local_waypoints_valid"],
            normal["trajectory.current_eef_xyz_m"],
        )
        anomaly_gain, features = gate(
            anomalous["trajectory.local_waypoints_xyz_m"],
            anomalous["trajectory.local_waypoints_valid"],
            anomalous["trajectory.current_eef_xyz_m"],
        )
        self.assertTrue(torch.all(anomaly_gain < normal_gain))
        self.assertTrue(torch.all((features >= 0.0) & (features <= 1.0)))

    def test_e2_uses_same_train_and_inference_wrapper(self):
        config = TrajectoryConditioningConfig(
            num_waypoints=4, gate_mode="geometry", encoder_hidden_dim=12
        )
        wrapper = SmolVLATrajectoryWrapper(TinySmolVLAPolicy(), config)
        batch = make_tiny_batch(num_waypoints=4, anomalous=True)
        loss, _ = wrapper(batch)
        actions = wrapper.predict_action_chunk(batch)
        self.assertEqual(loss.ndim, 0)
        self.assertEqual(tuple(actions.shape), (2, 4, 7))
        diagnostics = wrapper.diagnostics()
        self.assertEqual(diagnostics["gate_mode"], "geometry")
        self.assertIsNotNone(diagnostics["gate_gain_not_probability"])


if __name__ == "__main__":
    unittest.main()
