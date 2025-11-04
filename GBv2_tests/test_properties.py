import numpy as np
import pytest
import sys, os
from hypothesis import given, settings, strategies as st


# Go up to the top-level repo directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from GBv2_tests.GBv2_helper import (
    projector_from_los, sky_basis_from_los, project_points_xyz_to_xyprime,
    embed_xyprime_to_plane, unit
)

# Python see "scripts" as a top-level package
from scripts.geometry_config import rotate_about_axis as R


# Generate random angles between values of [-pi, pi]
angles = st.floats(min_value=-np.pi, max_value=np.pi, allow_nan=False, allow_infinity=False)

# Generate random unit vectors, with norms of unit vector less than 1e-8 left out
vecs = st.tuples(
    st.floats(-1,1), st.floats(-1,1), st.floats(-1,1)
).filter(lambda v: np.linalg.norm(v) > 1e-8)


# Projection idempotency: A projection transformation should have the property P^2 = P
# We check for idempotency by ||P^2 - P|| < tolerance
@settings(deadline=None, max_examples=200)
@given(vecs)
def test_projector_idempotent(v):
    nhat = unit(np.array(v))
    P = projector_from_los(nhat)
    assert np.linalg.norm(P @ P - P) < 1e-10

 
# Right-handed property, north x east = nhat, note nh = nhat
@settings(deadline=None, max_examples=200)
@given(vecs)
def test_basis_right_handed(v):
    nhat = unit(np.array(v))
    north, east, nh = sky_basis_from_los(nhat)
    # Check orthonormal
    # Unit length
    assert np.allclose(np.linalg.norm(east), 1.0, atol=1e-12)
    assert np.allclose(np.linalg.norm(north), 1.0, atol=1e-12)

    # Orthogonality
    assert np.allclose(np.dot(east, north), 0.0, atol=1e-12)
    assert np.allclose(np.dot(east, nh), 0.0, atol=1e-12)
    assert np.allclose(np.dot(north, nh), 0.0, atol=1e-11)

    # Right-handed: north x east = nhat
    assert np.allclose(np.dot(np.cross(north, east), nh), 1.0, atol=1e-12)


# Translation property of the projected plane
@settings(deadline=None, max_examples=20)
@given(vecs)
def test_translation_invariance_simple(v):
    N = 100 # Number of points
    nhat = unit(np.array(v))
    norm_v = np.linalg.norm(v)
    pts = np.random.randn(N,3)*norm_v # Orders of magnitude of v
    xy0 = project_points_xyz_to_xyprime(pts, nhat)

    # Add a LOS translation: no change
    xy1 = project_points_xyz_to_xyprime(pts + 5*nhat, nhat)
    assert np.allclose(xy0, xy1, rtol=0, atol=1e-11)

    # Add a pure sky-plane translation: merely shifts by the same amount
    north, east, _ = sky_basis_from_los(nhat)
    d = np.random.rand()*norm_v*east + np.random.rand()*norm_v*north
    xy2 = project_points_xyz_to_xyprime(pts + d, nhat)
    assert np.allclose(xy0 + [np.dot(d, east), np.dot(d, north)], xy2, atol=1e-11)
 

# Round-trip (reversible up to depth)
@settings(deadline=None, max_examples=10)
@given(vecs)
def test_roundtrip_plane_component(v):
    N = 100 # Number of points
    nhat = unit(v)
    pts = np.random.randn(N,3)
    xy = project_points_xyz_to_xyprime(pts, nhat)
    plane = embed_xyprime_to_plane(xy, nhat)
    # plane should equal pts with LOS component removed
    north, east, _ = sky_basis_from_los(nhat)
    proj = (pts @ east)[:,None]*east + (pts @ north)[:,None]*north
    assert np.allclose(plane, proj, atol=1e-10)


@settings(deadline=None, max_examples=50)
@given(st.floats(min_value=0.0, max_value=np.pi, allow_infinity=False, allow_nan=False))
def test_circular_orbit_axis_ratio(i):
    # Build a radius-1 circle in XY plane centered at origin
    t = np.linspace(0, 2*np.pi, 200, endpoint=False)
    circle = np.column_stack([np.cos(t), np.sin(t), np.zeros_like(t)])
    # LOS is orbital plane normal tilted by inclination i from +z around x-axis
    # Take base nhat0 = +z, rotate around x by i
    nhat = np.array([0, 0, 1.0])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(i), -np.sin(i)],
                   [0, np.sin(i),  np.cos(i)]])
    nhat = unit(Rx @ nhat)
 
    xy = project_points_xyz_to_xyprime(circle, nhat)
    # Axis lengths from covariance eigenvalues (proportional to squares)
    C = np.cov(xy.T)
    w, _ = np.linalg.eigh(C)
    w = np.sort(w)[::-1]
    ratio = np.sqrt(w[1]/w[0])  # b/a
    assert np.isclose(ratio, abs(np.cos(i)), atol=2e-2)

# Test the rodrigues rotation matrix can add by added when rotating about the same axis
@settings(deadline=None, max_examples=10)
@given(vecs)
def test_rodrigues_with_same_axis(v):
    rng = np.random.default_rng()
    th1, th2 = rng.uniform(-np.pi, np.nextafter(np.pi, np.inf), size = 2) # Range [-pi, pi]
    v = unit(v)
    R1 = R(v, th1)
    R2 = R(v, th2)
    R12 = R2 @ R1
    Rsum = R(v, th1+th2)
    assert np.allclose(R12, Rsum, atol=1e-12)

# Test the rotation helper function is non commutative, so it should fail when tried
@settings(deadline=None, max_examples=10)
@pytest.mark.xfail(reason="General Ω,i,ω do not compose by naive angle addition (non-commutative).")
def test_rodrigues_sum_not_valid():
    rng = np.random.default_rng()
    th1, th2, th3, th4, th5, th6 = rng.uniform(-np.pi, np.nextafter(np.pi, np.inf), size = 6) 
    # Demonstrate that naive sum fails in general
    x_axis = np.array([1,0,0])
    y_axis = np.array([0,1,0])
    z_axis = np.array([0,0,1])
    R1 = R(x_axis, th1) @ R(y_axis, th2) @ R(z_axis, th3)
    R2 = R(x_axis, th4) @ R(y_axis, th5) @ R(z_axis, th6) 
    R12 = R2 @ R1
    Rsum = R(x_axis, th1+th4) @ R(y_axis, th2+th5) @ R(z_axis, th3+th6)
    assert np.allclose(R12, Rsum, atol=1e-12)


@settings(deadline=None, max_examples=30)
@given(vecs)
def test_jacobian_sanity_finite_difference(v):
    """
    Numerically verify that the local Jacobian of the projection matches the
    analytical sky-basis rows.
    """

    # Unit line of sight
    nhat = unit(v)
    north, east, _ = sky_basis_from_los(nhat)

    # Analytical Jacobian: rows are east and north
    J_analytic = np.vstack([east, north])  # shape (2, 3)

    # Test at a random point in space
    r0 = np.random.randn(3)
    eps = 1e-6
    J_numeric = np.zeros((2, 3))

    # Finite-difference approximation
    for i in range(3):
        dr = np.zeros(3)
        dr[i] = eps
        xy_plus = project_points_xyz_to_xyprime(r0 + dr, nhat)
        xy_minus = project_points_xyz_to_xyprime(r0 - dr, nhat)
        J_numeric[:, i] = ((xy_plus - xy_minus) / (2 * eps)).ravel()

    # Assert numeric vs analytic
    assert np.allclose(J_numeric, J_analytic, atol=1e-10, rtol=1e-8)


if __name__ == "__main__":
     sys.exit(pytest.main([__file__, "-v"]))