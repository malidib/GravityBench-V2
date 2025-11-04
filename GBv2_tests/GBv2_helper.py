import numpy as np
import os
import sys
import glob
import pandas as pd
from tqdm import tqdm

from scripts.geometry_config import projection
import scripts.scenarios_config as scenarios_config


# Epsilon tolerance value
EPS = 1e-12
 
# Unit vector helper funciton, handle zero magnitude errors
def unit(v):
    """
    Return the unit vector of input. Checks for zero magnitude errors.
    """
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < EPS:
        raise ValueError("Zero-length vector")
    return v / n
  
# 3x3 projection matrix transformation helper function 
def projector_from_los(nhat):
    """
    Return 3x3 projector P onto plane orthogonal to nhat, which is the line of 
     sight (los), this is an orthographic.
    """
    nhat = unit(nhat)
    return np.eye(3) - np.outer(nhat, nhat)
 

def sky_basis_from_los(nhat):
    """
    Return (north, east, nhat) forming a right-handed orthonormal triad.
    North is the projection of global z=[0,0,1] into the sky plane (with fallback),
    East = nhat x North. 
    """
    nhat = unit(nhat)
    z = np.array([0.0, 0.0, 1.0])

    # If nhat ~ +/- z, fallback to y as 'north' source to avoid degeneracy
    # This is the case when the los points directly upwards along z, then take 
    # positive y as north

    if abs(np.dot(z, nhat)) > 1.0 - 1e-10:
        y = np.array([0.0, 1.0, 0.0])
        north = unit(y - np.dot(y, nhat) * nhat) 
    else:
        north = unit(z - np.dot(z, nhat) * nhat)
    east = unit(np.cross(nhat, north))

    # Ensure right-handed: north x east ≈ nhat
    if np.dot(np.cross(north, east), nhat) < 0:
        east = -east  # flip if needed
    return north, east, nhat
 
def project_points_xyz_to_xyprime(points_xyz, nhat, center=None):
    """
    Project Nx3 points to Nx2 using the V2 convention:
        x' = dot(r, East), y' = dot(r, North). Optionally subtract center first.
    """

    pts = np.asarray(points_xyz, dtype=float)
    if center is not None:
        pts = pts - np.asarray(center, dtype=float)
    north, east, _ = sky_basis_from_los(nhat)
    xprime = pts @ east
    yprime = pts @ north
    return np.column_stack([xprime, yprime])
 
def embed_xyprime_to_plane(xyprime, nhat):
    """Embed Nx2 back to 3D sky-plane: r_plane = x'*East + y'*North."""
    north, east, _ = sky_basis_from_los(nhat)
    xyprime = np.asarray(xyprime, dtype=float)
    return np.outer(xyprime[:,0], east) + np.outer(xyprime[:,1], north)
 
def mass_weighted_barycenter(r1, r2, m1, m2):
    """
    Return the position of barycenter mass.
    """
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    if not isinstance(m1, float) or not isinstance(m2, float):
        raise TypeError("Masses must be a floating numbers")
    return ((r1 * m1) + (r2 * m2))/ (m1 + m2)

def find_pairs_v2_csvs(V2_root=None):
    """Return list of (detailed_csv, projected_csv) path pairs under 
    scenarios/*_sims."""

    if V2_root is None:
        # Infer project root automatically from this file's location
        this_dir = os.path.dirname(__file__)
        V2_root = os.path.abspath(os.path.join(this_dir, ".."))  # one level up (project root)
    
    det_dir = os.path.join(V2_root, "scenarios", "detailed_sims")
    if not os.listdir(det_dir):
        raise FileNotFoundError("No detailed csv files, please generate some files.")
    proj_dir = os.path.join(V2_root, "scenarios", "projected_sims")
    detailed = sorted(glob.glob(os.path.join(det_dir, "*.csv")))
    print(detailed)
    projected = sorted(glob.glob(os.path.join(proj_dir, "*.csv")))
    stem_to_proj = {os.path.splitext(os.path.basename(p))[0]: p for p in projected}
#   stem_to_detailed = {os.path.splitext(os.path.basename(d))[0]: d for d in detailed}
    file_names = []
    
    for d in tqdm(detailed, desc="Checking detailed CSVs"):
        stem = os.path.splitext(os.path.basename(d))[0]
        # Generate the projection file if projection csv not found
        if stem not in stem_to_proj:
            print("INTERNAL: Projection csv file not found, creating projection files.")
            projection(pd.read_csv(f"{d}"), stem, save=True) # Create the projection dataframe
            proj_file = os.path.join(proj_dir, f"{stem}.csv")
            stem_to_proj[stem] = proj_file  # Update dict
        file_names.append((d, stem_to_proj[stem]))
    
    # I may not know the final answer units of the projected csv, so I am not doing projected to detailed

    #for p in tqdm(projected, desc="Checking projected CSVs"):
    #    stem = os.path.splitext(os.path.basename(p))[0]
    #    # Generate the detailed file if detailed csv not found
    #    if stem not in stem_to_detailed:
    #        print("INTERNAL: Detailed csv file not found, creating detailed files.")
    #        scenarios_config.variations[stem].create_binary(prompt="Base variations data", final_answer_units=('m', 's', 'kg'))
    #        det_file = os.path.join(det_dir, f"{stem}.csv")
    #        stem_to_detailed[stem] = det_file  # Update dict        
    #    file_names.append((stem_to_detailed[stem], p))

    return file_names

# Find the line of sight from a simulation data
def find_los(df):
    # Find the center of mass of the binary system
    # Get masses for COM calculation
    m1, m2 = df['star1_mass'].iloc[0], df['star2_mass'].iloc[0]
    total_mass = m1 + m2

    # Take the initial centre of mass as the reference point for the centre of projection, I could also take the mean of the centre of mass for better reference overall
    # Calculate COM coordinates
    COMx = (m1*df['star1_x'].iloc[0] + m2*df['star2_x'].iloc[0])/total_mass
    COMy = (m1*df['star1_y'].iloc[0] + m2*df['star2_y'].iloc[0])/total_mass
    COMz = (m1*df['star1_z'].iloc[0] + m2*df['star2_z'].iloc[0])/total_mass

    COM_vec = np.array([COMx, COMy, COMz])
    COM_unit_vec = unit(COM_vec) # Normal unit vector to the plane

    return COM_unit_vec, COM_vec