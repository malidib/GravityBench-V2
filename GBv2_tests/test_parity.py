import os
import numpy as np
import pandas as pd
import pytest
from dotenv import load_dotenv
load_dotenv(dotenv_path="GBv2_tests/GBv2.env", override=True)
import sys, re
import json

# Go up to the top-level repo directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

GBV1 = os.environ.get("GBv1_DIR")
GBV2 = os.environ.get("GBv2_DIR")

# Solar mass constant for conversions
Msun = 1.989e30  # kg

# Preconfigured testing scenario variations
json_path = os.path.join(GBV2, "GBv2_tests/parity_config.json")
    
with open(json_path, "r") as f:
    sample_variations = json.load(f)

@pytest.mark.parity
def test_v2_without_projection_matches_v1():
    if not (GBV1 and os.path.isdir(GBV1) and GBV2 and os.path.isdir(GBV2)):
        pytest.skip("Set both GBV1_DIR and GBV2_DIR to run parity test.")
    
    if sample_variations is None:
        pytest.fail("No variations to test on. Please add something to sample_variations.")
    
    BinaryScenario_v1 = import_binaryscenario(os.environ["GBv1_DIR"], tag="v1")
    BinaryScenario_v2 = import_binaryscenario(os.environ["GBv2_DIR"], tag="v2")

    for var_name, param in sample_variations.items():
        # Required parameters
        m1 = param.get("m1")*Msun
        m2 = param.get("m2")*Msun
        r1 = param.get("r1")
        r2 = param.get("r2")

        # Sanity check: all required fields exist
        if any(x is None for x in [var_name, m1, m2, r1, r2]):
            pytest.fail(f"Variation '{var_name}' missing one or more required keys: "
                "('name', 'm1', 'm2', 'r1', 'r2').")

        # Optional parameters
        e = param.get("ellipticity", 0.0)
        maxtime = param.get("maxtime", None)
        num_orbits = param.get("num_orbits", 10)
        proper_motion_direction = param.get("proper_motion_direction", None)
        proper_motion_magnitude = param.get("proper_motion_magnitude", 0.0)
        drag_tau = param.get("drag_tau", None)
        mod_gravity_exponent = param.get("mod_gravity_exponent", None)
        units = param.get("units", ("m", "s", "kg"))

        if units == ['yr', 'AU', 'Msun']:
            # Convert to meters
            AU = 1.496e11        # meters
            m1 /= Msun
            m2 /= Msun
            r1 = np.array(r1) / AU
            r2 = np.array(r2) / AU

        if units == ['s', 'cm', 'g']:
            cm = 100
            g = 1000
            m1 *= g
            m2 *= g
            r1 = np.array(r1) * cm
            r2 = np.array(r2) * cm

        v1 = BinaryScenario_v1(variation_name = var_name, star1_mass=m1, star2_mass=m2,
                               star1_pos=r1, star2_pos=r2, maxtime=maxtime, num_orbits=num_orbits, ellipticity=e, 
                               proper_motion_direction=proper_motion_direction, proper_motion_magnitude=proper_motion_magnitude, 
                               drag_tau=drag_tau, mod_gravity_exponent=mod_gravity_exponent, 
                               units=tuple(units))

        v2 = BinaryScenario_v2(variation_name = var_name, star1_mass=m1, star2_mass=m2,
                               star1_pos=r1, star2_pos=r2, maxtime=maxtime, num_orbits=num_orbits, ellipticity=e, 
                               proper_motion_direction=proper_motion_direction, proper_motion_magnitude=proper_motion_magnitude, 
                               drag_tau=drag_tau, mod_gravity_exponent=mod_gravity_exponent, 
                               units=tuple(units))

        
        # Forcefully regenerate files
        v1.create_binary(prompt="GBV1", final_answer_units=tuple(units)) 
        v2.create_binary(prompt="GBV2", final_answer_units=tuple(units))

        # Expect the same detailed_sims content for identical scenario/seed when projection=False
        v1_det_path = os.path.join(GBV1, "scenarios", "detailed_sims", f"{var_name}.csv")
        v2_det_path = os.path.join(GBV2, "scenarios", "detailed_sims", f"{var_name}.csv")

        assert os.path.exists(v1_det_path), f"File missing: {v1_det_path}"
        assert os.path.exists(v2_det_path), f"File missing: {v2_det_path}"  
 
        v1_df = pd.read_csv(v1_det_path)
        v2_df = pd.read_csv(v2_det_path)

        v1_df = v1_df[["time","star1_x","star1_y","star1_z","star2_x","star2_y","star2_z","star1_vx","star1_vy","star1_vz","star2_vx","star2_vy","star2_vz","star1_mass","star2_mass","separation","force","star1_accel","star2_accel","specific_angular_momentum","orbital_period","mean_motion","semimajor_axis","eccentricity","inclination","true_anomaly","mean_anomaly","time_of_pericenter_passage","radial_distance_from_reference"]]       
        v2_df = v2_df[["time","star1_x","star1_y","star1_z","star2_x","star2_y","star2_z","star1_vx","star1_vy","star1_vz","star2_vx","star2_vy","star2_vz","star1_mass","star2_mass","separation","force","star1_accel","star2_accel","specific_angular_momentum","orbital_period","mean_motion","semimajor_axis","eccentricity","inclination","true_anomaly","mean_anomaly","time_of_pericenter_passage","radial_distance_from_reference"]]

        if not np.allclose(v1_df, v2_df, atol=1e-8):
            pytest.fail(f"Parity mismatch for variation: {var_name}")


import importlib.util
def import_binaryscenario(repo_path, tag):
    """Import BinaryScenario and its dependencies under an isolated namespace."""
    sys.path.insert(0, repo_path)
    try:
        # Load the repo's Binary module
        binary_path = os.path.join(repo_path, "generalscenarios", "Binary.py")
        binary_spec = importlib.util.spec_from_file_location(
            f"generalscenarios_{tag}.Binary",
            binary_path
        )
        binary_module = importlib.util.module_from_spec(binary_spec)
        binary_spec.loader.exec_module(binary_module)

        # Load the repo's scenarios_config under a unique name
        scen_path = os.path.join(repo_path, "scripts", "scenarios_config.py")
        scen_spec = importlib.util.spec_from_file_location(
            f"scenarios_config_{tag}",
            scen_path
        )
        scen_module = importlib.util.module_from_spec(scen_spec)

        # Inject the correct Binary *before* execution
        scen_module.Binary = binary_module.Binary
        sys.modules[f"generalscenarios.Binary"] = binary_module  # temporarily override import resolution
        scen_spec.loader.exec_module(scen_module)

        return scen_module.BinaryScenario

    finally:
        sys.path.pop(0)

if __name__ == "__main__":
     sys.exit(pytest.main([__file__, "-v", "--strict-integration"]))