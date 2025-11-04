import numpy as np
import pandas as pd
import pytest
import sys
import os
from tqdm import tqdm

# Go up to the top-level repo directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from GBv2_tests.conftest import add_repo_to_syspath

# Setup repo root path, allows other imports to work
repo_root = add_repo_to_syspath()

from GBv2_tests.GBv2_helper import (project_points_xyz_to_xyprime, unit, find_pairs_v2_csvs)

# Checking barycenter consistent with projection, projection a linear function
@pytest.mark.integration
def test_barycenter_consistent_in_projection(request):
    # Find the detailed and projected csv files, generate them if they one of
    # the files are missings.
    GBv2_path = os.environ.get("GBv2_DIR")
    assert GBv2_path and os.path.isdir(GBv2_path), "Set GBv2_DIR to V2 repo."
    pairs = find_pairs_v2_csvs(GBv2_path)
    if not pairs:
        if request.config.getoption("--strict-integration"):
            pytest.fail("No CSV pairs found.")
        pytest.skip("No CSV pairs found; generate some detailed csv files.")
 
    # Check barycenter consistency for each pair
    for det_path, proj_path in tqdm(pairs, desc="Checking for barycenter of every pair"):
        det = pd.read_csv(det_path)
        proj = pd.read_csv(proj_path)
        m1 = float(det["star1_mass"].iloc[0])
        m2 = float(det["star2_mass"].iloc[0])
        r1_detailed = np.column_stack([det["star1_x"], det["star1_y"], det["star1_z"]])
        r2_detailed = np.column_stack([det["star2_x"], det["star2_y"], det["star2_z"]])
        
        # 3D barycenter per timestep -> project
        com3d = (m1*r1_detailed + m2*r2_detailed) / (m1+m2)  # Nx3
        nhat = unit(com3d[0]) # Get the initial line of sight
        com2d_ref = project_points_xyz_to_xyprime(com3d, nhat)  # Nx2
    
        # 2D barycenter from projected CSV
        r1_projected = np.column_stack([proj["star1_x"], proj["star1_y"]]) # Note z = 0
        r2_projected = np.column_stack([proj["star2_x"], proj["star2_y"]])
        com2d_v2 = (m1*r1_projected + m2*r2_projected) / (m1+m2)

        norm_los = np.linalg.norm(com3d[0])
    
        assert np.allclose(com2d_v2, com2d_ref, atol=1e-15*norm_los)

# Checking projection csv is correct
@pytest.mark.integration
def test_v2_csv_projection_matches_reference(request):
    GBv2_path = os.environ.get("GBv2_DIR")
    assert GBv2_path and os.path.isdir(GBv2_path), "Set GBv2_DIR to V2 repo."
    pairs = find_pairs_v2_csvs(GBv2_path)
    if not pairs:
        if request.config.getoption("--strict-integration"):
            pytest.fail("No detailed/projected CSV pairs found under scenarios/.")
        pytest.skip("No CSV pairs found; run V2 simulation with projection enabled.")
    # Validate first few pairs
    for detailed_path, projected_path in tqdm(pairs, desc="Checking each projection csv matches"):
        det = pd.read_csv(detailed_path)
        proj = pd.read_csv(projected_path)

        # Build per-timestep initial COM LOS from the *initial* positions
        r1_detailed = np.column_stack([det["star1_x"], det["star1_y"], det["star1_z"]])
        r2_detailed = np.column_stack([det["star2_x"], det["star2_y"], det["star2_z"]])
        r10 = r1_detailed[0]
        r20 = r2_detailed[0]
        m1 = float(det["star1_mass"].iloc[0])
        m2 = float(det["star2_mass"].iloc[0])
        com0 = (m1*r10 + m2*r20) / (m1+m2)
        nhat = unit(com0)

        # Project both bodies independently and compare against CSV (allow constant 2D offset)
        r1p_ref = project_points_xyz_to_xyprime(r1_detailed, nhat)  # Nx2
        r2p_ref = project_points_xyz_to_xyprime(r2_detailed, nhat)  # Nx2

        r1p_v2 = np.column_stack([proj["star1_x"], proj["star1_y"]])
        r2p_v2 = np.column_stack([proj["star2_x"], proj["star2_y"]])
        
        # Allow an overall 2D translation (some pipelines center differently).
        # Fit offset by first row:
        off1 = r1p_v2[0] - r1p_ref[0]
        off2 = r2p_v2[0] - r2p_ref[0]
        assert np.allclose(off1, off2, atol=1), "Inconsistent per-body translation"
        assert np.allclose(r1p_v2, r1p_ref + off1, atol=1e-5)
        assert np.allclose(r2p_v2, r2p_ref + off1, atol=1e-5)


if __name__ == "__main__":
     sys.exit(pytest.main([__file__, "-v"]))