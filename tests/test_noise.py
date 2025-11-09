import os
import sys
import unittest

import numpy as np
import pandas as pd

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.scenarios_config import BinaryScenario, calculate_orbital_momentum, Msun  # noqa: E402


class TestNoiseGeneration(unittest.TestCase):
    """Validate that enabling noise perturbs simulation outputs."""

    variation_name = "unit_test_noise_scenario"

    def setUp(self):
        self._cleanup_files()

    def tearDown(self):
        self._cleanup_files()

    def _cleanup_files(self):
        for folder in ["scenarios/sims", "scenarios/detailed_sims"]:
            path = os.path.join(folder, f"{self.variation_name}.csv")
            if os.path.exists(path):
                os.remove(path)

    def _base_scenario(self, **kwargs):
        star1_mass = 3.0 * Msun
        star2_mass = 1.5 * Msun
        star1_pos = np.array([-5e8, 0.0, 0.0])
        star2_pos = np.array([5e8, 0.0, 0.0])

        # Reuse orbital momentum helper to ensure a stable orbit
        star1_mom, star2_mom = calculate_orbital_momentum(
            star1_mass,
            star2_mass,
            star1_pos,
            star2_pos,
        )

        scenario = BinaryScenario(
            variation_name=self.variation_name,
            star1_mass=star1_mass,
            star2_mass=star2_mass,
            star1_pos=star1_pos,
            star2_pos=star2_pos,
            maxtime=None,
            num_orbits=1,
            ellipticity=0.0,
            proper_motion_direction=None,
            proper_motion_magnitude=0.0,
            drag_tau=None,
            mod_gravity_exponent=None,
            units=('m', 's', 'kg'),
            projection=True,
            **kwargs,
        )

        # Overwrite precomputed momenta with the stable orbit values
        scenario.star1_momentum = star1_mom
        scenario.star2_momentum = star2_mom
        return scenario

    def test_noise_changes_positions_and_is_reproducible(self):
        # Baseline run without noise
        clean_scenario = self._base_scenario(enable_noise=False, noise_seed=1234)
        clean_binary = clean_scenario.create_binary(prompt="", final_answer_units=None, skip_simulation=False)
        clean_path = os.path.join("scenarios/sims", f"{clean_binary.filename}.csv")
        clean_df = pd.read_csv(clean_path)

        # Run with noise enabled
        noisy_scenario = self._base_scenario(
            enable_noise=True,
            noise_type='gaussian',
            noise_level=0.1,
            noise_seed=1234,
        )
        noisy_binary = noisy_scenario.create_binary(prompt="", final_answer_units=None, skip_simulation=False)
        noisy_path = os.path.join("scenarios/sims", f"{noisy_binary.filename}.csv")
        noisy_df = pd.read_csv(noisy_path)

        self.assertFalse(noisy_df[['star1_x', 'star1_y', 'star1_z', 'star2_x', 'star2_y', 'star2_z']].equals(
            clean_df[['star1_x', 'star1_y', 'star1_z', 'star2_x', 'star2_y', 'star2_z']]
        ), "Noise-enabled simulation should differ from noise-free baseline.")

        # Noise should be reproducible for the same seed
        repeat_scenario = self._base_scenario(
            enable_noise=True,
            noise_type='gaussian',
            noise_level=0.1,
            noise_seed=1234,
        )
        repeat_binary = repeat_scenario.create_binary(prompt="", final_answer_units=None, skip_simulation=False)
        repeat_path = os.path.join("scenarios/sims", f"{repeat_binary.filename}.csv")
        repeat_df = pd.read_csv(repeat_path)

        pd.testing.assert_frame_equal(noisy_df, repeat_df)


if __name__ == "__main__":
    unittest.main()
