import numpy as np
import pytest
import sys, os, json
import pandas as pd

# Go up to the top-level repo directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from GBv2_tests.GBv2_helper import (
    projector_from_los, sky_basis_from_los, project_points_xyz_to_xyprime,
    embed_xyprime_to_plane, unit, find_pairs_v2_csvs, find_los
)

from scripts.geometry_config import rotate_about_axis as R

def test_projector_idempotency_with_simulations():
    detailed_paths = find_pairs_v2_csvs()
    for path in detailed_paths:
        # Read csv
        df = pd.read_csv(path[0])

        # Find the center of mass of the binary system
        COM_unit_vec, COM_vec = find_los(df)

        star1_coord = np.vstack([df['star1_x'], df['star1_y'], df['star1_z']])  # (3, N)
        star2_coord = np.vstack([df['star2_x'], df['star2_y'], df['star2_z']])

        # Translate so initial COM is at the origin
        centered1 = star1_coord - COM_vec[:, None]   # (3, N)
        centered2 = star2_coord - COM_vec[:, None]
        
        P = np.eye(3) - np.outer(COM_unit_vec, COM_unit_vec)
        assert np.allclose(P @ P, P, atol=1e-12)

        v_proj1 = centered1 - (COM_unit_vec @ centered1) * COM_unit_vec[:, None]
        v_proj2 = centered2 - (COM_unit_vec @ centered2) * COM_unit_vec[:, None]

        norm_COM = np.linalg.norm(COM_vec)

        assert np.allclose(0, COM_unit_vec @ v_proj1, atol=norm_COM*1e-14)
        assert np.allclose(0, COM_unit_vec @ v_proj2, atol=norm_COM*1e-14)

def test_translation_invariance_with_simulations():
    detailed_paths = find_pairs_v2_csvs()
    for path in detailed_paths:
        # Read csv
        df = pd.read_csv(path[0])
    
        COM_unit_vec, COM_vec = find_los(df)

        star1_coord = np.column_stack((df['star1_x'], df['star1_y'], df['star1_z']))
        star2_coord = np.column_stack((df['star2_x'], df['star2_y'], df['star2_z']))
        xy1 = project_points_xyz_to_xyprime(star1_coord, COM_unit_vec)
        xy2 = project_points_xyz_to_xyprime(star2_coord, COM_unit_vec)

        alpha = np.random.rand()*np.linalg.norm(COM_vec)
        beta = np.random.rand()*np.linalg.norm(COM_vec)

        # Add a LOS translation: no change
        xy1_translated = project_points_xyz_to_xyprime(star1_coord + alpha*COM_unit_vec, COM_unit_vec)
        xy2_translated = project_points_xyz_to_xyprime(star2_coord + beta*COM_unit_vec, COM_unit_vec)
        assert np.allclose(xy1, xy1_translated, rtol=1e-8, atol=0.0)
        assert np.allclose(xy2, xy2_translated, rtol=1e-8, atol=0.0)

        # Add a pure sky-plane translation: merely shifts by the same amount
        north, east, _ = sky_basis_from_los(COM_unit_vec)
        d = alpha*east + beta*north
        xy3 = project_points_xyz_to_xyprime(star1_coord + d, COM_unit_vec)
        xy4 = project_points_xyz_to_xyprime(star2_coord + d, COM_unit_vec)
        assert np.allclose(xy1 + [np.dot(d, east), np.dot(d, north)], xy3, atol=1e-11)
        assert np.allclose(xy2 + [np.dot(d, east), np.dot(d, north)], xy4, atol=1e-11)

def test_jacobian_sanity_finite_difference_with_simulations():
    """
    Numerically verify that the local Jacobian of the projection matches the
    analytical sky-basis rows.
    """
    detailed_paths = find_pairs_v2_csvs()

    for path in detailed_paths:

        # Read csv
        df = pd.read_csv(path[0])

        # Find the center of mass of the binary system
        COM_unit_vec, COM_vec = find_los(df)

        north, east, _ = sky_basis_from_los(COM_unit_vec)

        # Analytical Jacobian: rows are east and north that are normalised, so we can set rtol to be lower
        J_analytic = np.vstack([east, north])  # shape (2, 3)

        # Test all the points
        star1_coord = np.column_stack((df['star1_x'], df['star1_y'], df['star1_z']))
        star2_coord = np.column_stack((df['star2_x'], df['star2_y'], df['star2_z']))
        eps = 1e-10*np.linalg.norm(COM_vec)
        J_numeric_1 = np.zeros((2, 3))
        J_numeric_2 = np.zeros((2, 3))

        # Finite-difference approximation
        for k in np.linspace(0, star1_coord.shape[1], 10000, dtype=int):
            for i in range(3):
                dr = np.zeros(3)
                dr[i] = eps
                star1_xy_plus  = project_points_xyz_to_xyprime(star1_coord[k] + dr, COM_unit_vec)
                star1_xy_minus = project_points_xyz_to_xyprime(star1_coord[k] - dr, COM_unit_vec)
                star2_xy_plus  = project_points_xyz_to_xyprime(star2_coord[k] + dr, COM_unit_vec)
                star2_xy_minus = project_points_xyz_to_xyprime(star2_coord[k] - dr, COM_unit_vec)
                J_numeric_1[:, i] = (star1_xy_plus - star1_xy_minus) / (2 * eps)
                J_numeric_2[:, i] = (star2_xy_plus - star2_xy_minus) / (2 * eps)

            # Assert numeric vs analytic
            assert np.allclose(J_numeric_1, J_analytic, atol=1e-4, rtol=1e-4)
            assert np.allclose(J_numeric_2, J_analytic, atol=1e-4, rtol=1e-4)




if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))