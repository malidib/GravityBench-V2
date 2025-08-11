import numpy as np
import pandas as pd

"""
Argument of periapsis, as included in rebound, calculataed as the angle between the longitude of ascending node and eccentricity vector. Angle are in radians
"""

class Scenario:
    def __init__(self, scenario_creator, skip_simulation=False):
        self.scenario_creator = scenario_creator

        prompt = """Determine the argument of periapsis of the system's orbit. Take the longitude of ascending node as reference."""
        final_answer_units = "rad"

        self.binary_sim = self.scenario_creator.create_binary(prompt, final_answer_units, skip_simulation=skip_simulation)

    def true_answer(self, N_obs=None, verification=False, return_empirical=False) -> float:
        """
        Return the true answer for the environment.
        
        Args:
            N_obs: Number of observations to use (if None, use all)
            verification: Whether to verify values match
            return_empirical: If True, return the empirically derived value;
                              if False, return the value inputted into the simulation or using Rebound simulated details typically hidden
        """
        # Load the simulation data
        df = pd.read_csv(f"scenarios/detailed_sims/{self.binary_sim.filename}.csv")
        
        if N_obs is not None:
            indices = np.linspace(0, len(df) - 1, N_obs)
            df = df.iloc[indices].reset_index(drop=True)

        # Get mass and reduced mass
        m1, m2 = df['star1_mass'].iloc[0], df['star2_mass'].iloc[0]
        total_mass = m1 + m2
        reduced_mass = (m1 * m2)/total_mass

        # Calculate the eccentricity vector
        r_rel = np.stack([
            df['star2_x'] - df['star1_x'],
            df['star2_y'] - df['star1_y'],
            df['star2_z'] - df['star1_z']
            ], axis=1)
        
        v_rel = np.stack([
            df['star2_vx'] - df['star1_vx'],
            df['star2_vy'] - df['star1_vy'],
            df['star2_vz'] - df['star1_vz']
            ], axis=1)  

        # Calculate the specific angular momentum vector
        h_vec = np.cross(r_rel, v_rel)
        h_avg = h_vec.mean(axis=0)
        h_unit = h_avg / np.linalg.norm(h_avg)
        longitude_of_ascending_node_vector =  np.cross([0, 0, 1], h_unit)

        # Calculate the eccentricity vector
        reduced_mass = (m1 * m2)/total_mass # Reduced mass of the binary system
        r_norm = np.linalg.norm(r_rel, axis=1).reshape(-1, 1)
        eccentricity_vector = np.mean((np.cross(v_rel, h_vec) / reduced_mass) - (r_rel / r_norm), axis=0)

        # Calculate the argument of periapsis
        norm_e = np.linalg.norm(eccentricity_vector)
        norm_long = np.linalg.norm(longitude_of_ascending_node_vector)

        if norm_e < 1e-12 or norm_long <1e-12:
            argument_of_periapsis = 0.0 # e≈0 or Ω undefined, handle near-zero norms
        else:
            cosine = np.dot(eccentricity_vector, longitude_of_ascending_node_vector) / (norm_e * norm_long)
            cosine = np.clip(cosine, -1.0, 1.0) # Clamp cosine-argument to [-1 , 1]
            argument_of_periapsis = np.arccos(cosine)

            # sin() disambiguation
            sin_argp = np.dot(np.cross(longitude_of_ascending_node_vector, eccentricity_vector), h_unit)
            # Flip angle if sin is negative
            if sin_argp < 0:
                argument_of_periapsis = 2 * np.pi - argument_of_periapsis

        # verification, inclinaiton is usually constant throughout the simulation, so we can use the first value
        arg_rebound = df['argument_of_periapsis'].iloc[0]
        if verification:
            if arg_rebound == 0:
                assert float(argument_of_periapsis) == 0.0, f"rebound argument of periapsis is 0, but calculated argument is not 0"
            else:
                assert abs(argument_of_periapsis - arg_rebound) < 0.01 * arg_rebound, f"{argument_of_periapsis} and {arg_rebound} are not within 1% of each other"

        # If return empirical
        if return_empirical:
            return argument_of_periapsis  # Return the calculated inclination if empirical value is requested
        else:
            return arg_rebound  # Return the rebound calculated inclination if not requesting empirical value
        
