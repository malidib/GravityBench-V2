import pytest
import sys, os, json
import numpy as np
import pandas as pd
import rebound
import ast

# Go up to the top-level repo directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.geometry_config import reset
import scripts.scenarios_config as scenarios_config

def test_random_geometry_accuracy():
    GBV2 = os.environ.get("GBv2_DIR")

    # Preconfigured testing scenario variations
    json_path = os.path.join(GBV2, "GBv2_tests/parity_config.json")
        
    with open(json_path, "r") as f:
        sample_variations = json.load(f)
    
    if not GBV2 and os.path.isdir(GBV2):
        pytest.skip("Set GBV2_DIR to run parity test.")
    
    if sample_variations is None:
        pytest.fail("No variations to test on. Please add something to sample_variations.")

    for var_name, param in sample_variations.items():
        # It does the assertion automatically here
        geometry(file_name=var_name, random=True, translation=False)

    # Once done remove all transformed variations
    reset()



# To reduce computation, we ignore the entire dataframe verification step
def geometry(file_name:str, random=False, translation=False):
    """
    If random, randomly transform the geometry of the binary system from a orientation to an another random orientation. It will account for
    any original orientation of the binary system, and transform it to a new orientation.
    
    This is done in four steps:
    1. Randomly translate the binary system x,y,z. The range of translation is restricted between (-COM, COM) in each perpendicular direction, 
    where COM is the center of mass of the binary system.
    2. Randomly rotate the binary system about the y-axis of the COM by a random inclination angle. [0, pi]
    3. Randomly rotate the binary system about the z-axis of the COM by a random longitude of ascending node. [-pi, pi]
    4. Randomly rotate the binary system about the normal axis of the orbital plane by a random argumet of periapsis. [0, 2*pi]

    If not random, it will transform the geometry of the binary system from a orientation to another orientation specified in the file_name.
    The steps are similar to the random case, but with the parameters specified.

    For not random, the file name must be specified as follows: "9.6 M, 3.1 M; Inc_0.5_Long_0.2_Arg 0.1; Trans_[4e10, -3e4, 6e10]". This is configured in scenarios_config.json file,
    with only the base variation needed "9.6 M, 3.1 M" in the scenarios_config.py

    Any parameters that is wanted left unchanged can be left out, E.g. "9.6 M, 3.1 M; Inc_0.5_Arg 0.1" will leave longitude, and translation unchanged

    Note: If we would like to use original dataframe, we will have reset momentums as well, but it should be relatively easy as we already have velocity
    transformed. The transformation of velocity is necessary to recalculate specific angular momentum for rotation matrix.

    Parameters:
    -----------
    file_name : str
        Name of the original variation in detailed_sims
    verification : bool, optional
        Whether to verify results (default False)
    random : bool, optional
        Whether to intiate random geometry

    Returns:
    --------
    str
        The name of the file containing the transformed geometry. 
        It is named as follows {file_name}; Inc_{inclination_angle}_Long_{longitude of ascending node}_Arg_{argument of periapsis}; Trans_[x, y, z]
    """

    # Check the filename whether the base variations exists, if not simulate the variation
    col = [' Inc', 'Long', 'Arg', 'Trans']

    if any(sub in file_name for sub in col):
        base_variation_name = file_name.split(';')[0]
    else:
        base_variation_name = file_name

    # Determine if the base simulation should be skipped
    sim_csv_file_path = f"scenarios/sims/{base_variation_name}.csv"
    detailed_sim_csv_file_path = f"scenarios/detailed_sims/{base_variation_name}.csv"

    skip_simulation = os.path.exists(sim_csv_file_path) and os.path.exists(detailed_sim_csv_file_path)

    if not skip_simulation:
        if base_variation_name not in scenarios_config.variations:
            raise ValueError("INTERNAL: Base variations not found in variations in scenario_config.py. Please update.")
        else:
            print("INTERNAL: Base variations csv file not found, creating simulation data.")
            scenarios_config.variations[base_variation_name].create_binary(prompt="Base variations data", final_answer_units=('m', 's', 'kg'))

    # Read the csv file
    df = pd.read_csv(detailed_sim_csv_file_path)
    df = df.astype('float64')

    # Find the center of mass of the binary system
    # Get masses for COM calculation
    m1, m2 = df['star1_mass'].iloc[0], df['star2_mass'].iloc[0]
    total_mass = m1 + m2

    # Calculate COM coordinates
    df['COMx'] = (m1*df['star1_x'] + m2*df['star2_x'])/total_mass
    df['COMy'] = (m1*df['star1_y'] + m2*df['star2_y'])/total_mass
    df['COMz'] = (m1*df['star1_z'] + m2*df['star2_z'])/total_mass

    # Check if random orientation is wanted
    if random:
        COMx = df['COMx'].mean() * 1e-6
        COMy = df['COMy'].mean() * 1e-6
        COMz = df['COMz'].mean() * 1e-6
        if translation == True:
            # Random translation in x, y, z with range from (-COM, COM)
            translation_x = np.random.uniform(-COMx, COMx)
            translation_y = np.random.uniform(-COMy, COMy)
            translation_z = np.random.uniform(-COMz, COMz)
        else:
            # Else set it to zero
            translation_x = 0.0
            translation_y = 0.0
            translation_z = 0.0

        # Random inclination about the xy plane, longitude of ascending node about positive x-axis, and argument of periapsis
        inclination = np.random.uniform(0, np.pi)  # Random inclination between 0 and pi
        longitude_of_ascending_node = np.random.uniform(-np.pi, np.pi)  # Random longitude of ascending node between -pi and pi, with positive x-axis as reference
        argument_of_periapsis = np.random.uniform(0, 2*np.pi) # Random argument of periapsis between 0 and 2pi
    else:
        # Configuring the parameter data through the name
        file_split = file_name.split(';')
        if 'Trans' in file_name:
            translation = file_split[-1].split('_')
            translation_list = ast.literal_eval(translation[-1])
            translation_x, translation_y, translation_z = float(translation_list[0]), float(translation_list[1]), float(translation_list[2])
            geometry_name = file_split[-2].strip().split('_')
        else: 
            geometry_name = file_split[-1].strip().split('_')
            translation_x = 0.0
            translation_y = 0.0
            translation_z = 0.0

        index = 1
        # By following the order Inc, Long, Arg, the index will correctly correspond to whichever is wanted
        if 'Inc' in geometry_name:
            inclination = float(geometry_name[index])
            index += 2
        else:
            inclination = df['inclination'].iloc[0]

        if 'Long' in geometry_name:
            longitude_of_ascending_node = float(geometry_name[index])
            index += 2
        else:
            longitude_of_ascending_node = df['longitude_of_ascending_node'].iloc[0]

        if 'Arg' in geometry_name:
            argument_of_periapsis = float(geometry_name[index])
        else:
            argument_of_periapsis = df['argument_of_periapsis'].iloc[0]

    # Check if values are in correct bounds
    if inclination < 0 or inclination > np.pi:
        raise ValueError(f"Inclination angle out of bounds, it is inputed as {inclination}. Try to keep it within [0, pi]")
    if longitude_of_ascending_node < -np.pi or longitude_of_ascending_node > np.pi:
        raise ValueError(f"Longitude of ascending node out of bounds, it is inputed as {longitude_of_ascending_node}. Try to keep it within [-pi, pi]")
    if argument_of_periapsis < 0 or argument_of_periapsis > 2*np.pi:
        raise ValueError(f"Argument of periapsis out of bounds, it is inputed as {argument_of_periapsis}. Try to keep it within [0, 2*pi]")

    # Apply translation to positions
    df['star1_x'] += translation_x
    df['star1_y'] += translation_y
    df['star1_z'] += translation_z
    df['star2_x'] += translation_x
    df['star2_y'] += translation_y
    df['star2_z'] += translation_z
    df['COMx'] += translation_x
    df['COMy'] += translation_y
    df['COMz'] += translation_z

    translation = np.array([translation_x, translation_y, translation_z])

    # Get the current inclination, longitude of ascending node, and argument of periapsis
    current_inclination = df['inclination'].iloc[0]
    current_longitude_of_ascending_node = df['longitude_of_ascending_node'].iloc[0]
    current_argument_of_periapsis = df['argument_of_periapsis'].iloc[0]

    # Get Gravitational constant
    units = scenarios_config.variations[base_variation_name].units
    if units == ('yr', 'AU', 'Msun'):
        # Astronomical units: AU^3 / (Msun * yr^2)
        G = 4 * np.pi**2
    elif units == ('s', 'cm', 'g'):
        # CGS units: cm^3 / (g * s^2)
        G = 6.67430e-8  
    else:  # Default SI units
        # SI units: m^3 / (kg * s^2)
        G = 6.67430e-11

    # Find the original star position for the variation setup
    star1_pos = np.array(scenarios_config.variations[base_variation_name].star1_pos).astype('float64')
    star2_pos = np.array(scenarios_config.variations[base_variation_name].star2_pos).astype('float64')

    # Apply translation to COM_vec
    star1_pos += translation
    star2_pos += translation

    # Find the COM vector
    COM_vec = np.array([(m1*star1_pos[0] + m2*star2_pos[0])/total_mass, (m1*star1_pos[1] + m2*star2_pos[1])/total_mass, (m1*star1_pos[2] + m2*star2_pos[2])/total_mass])

    # Call the necessary parameters from the variations base setup
    modified_exponent = scenarios_config.variations[base_variation_name].mod_gravity_exponent or 2
    ellipticity = scenarios_config.variations[base_variation_name].ellipticity
    proper_motion_direction = scenarios_config.variations[base_variation_name].proper_motion_direction
    proper_motion_magnitude = scenarios_config.variations[base_variation_name].proper_motion_magnitude

    # Calculate base orbital velocity
    relative_position = star2_pos - star1_pos
    r = np.linalg.norm(relative_position)
    orbital_velocity = np.sqrt(G * total_mass / r**(modified_exponent - 1))

    # Determine orbital plane direction
    if relative_position[0] != 0 or relative_position[1] != 0:
        velocity_direction = np.cross(relative_position, [0, 0, 1])  # z-axis
    else:
        velocity_direction = np.cross(relative_position, [0, 1, 0])  # y-axis
    velocity_direction = velocity_direction.astype(float)
    velocity_direction /= np.linalg.norm(velocity_direction)

    # Add eccentricity component
    radial_direction = relative_position / r
    velocity_radial = radial_direction * (orbital_velocity * ellipticity)
    velocity_total = velocity_direction * orbital_velocity + velocity_radial

    # Mass-weighted velocity distribution
    star2_velocity = np.array(velocity_total * (m1 / total_mass))
    star1_velocity = np.array(-velocity_total * (m2 / total_mass))

    if scenarios_config.variations[base_variation_name].proper_motion_magnitude is not None:
        # Add proper motion velocity component
        if proper_motion_magnitude > 0:
            if proper_motion_direction is None:
                proper_motion_dir = np.array([1.0, 1.0, 0.0])
            else:
                proper_motion_dir = np.array(proper_motion_direction, dtype=float)
            proper_motion_dir /= np.linalg.norm(proper_motion_dir)
            proper_motion_velocity = proper_motion_dir * proper_motion_magnitude
            df[['star1_vx', 'star1_vy', 'star1_vz']] -= proper_motion_velocity
            df[['star2_vx', 'star2_vy', 'star2_vz']] -= proper_motion_velocity

    # Apply inclination using Rodrigues' rotation matrix
    # Find the relative position and velocity
    r_rel = star2_pos - star1_pos
    v_rel = star2_velocity - star1_velocity

    # Find specific angular momentum unit vector
    h_vec = np.cross(r_rel, v_rel)
    h_unit = h_vec / np.linalg.norm(h_vec)

    # Find the inclination rotation axis
    inclination_rotation_axis = np.cross([0, 0, 1], h_unit)
    if np.linalg.norm(inclination_rotation_axis) < 1e-12 and (h_unit[2] > 0):
        inclination_rotation_axis = np.cross(COM_vec, [0, 0, 1])
    elif (np.linalg.norm(inclination_rotation_axis) < 1e-12) and (h_unit[2] < 0):
        inclination_rotation_axis = np.cross(COM_vec, [0, 0, -1])

    # If COM_vec parallel to the z-axis, we consider a fall-back
    if (np.linalg.norm(inclination_rotation_axis) < 1e-12) and (h_unit[2] > 0):
        inclination_rotation_axis = np.array([0, -1, 0])
    elif (np.linalg.norm(inclination_rotation_axis) < 1e-12) and (h_unit[2] < 0):
        inclination_rotation_axis = np.array([0, 1, 0])

    # Get inclination difference
    inc_diff = inclination - current_inclination

    # Perform the inclination rotation along the eccentricity vector
    R_inc = rotate_about_axis(inclination_rotation_axis, inc_diff)

    rel_star1 = star1_pos - COM_vec  # Relative position of star1 from COM
    rel_star2 = star2_pos - COM_vec  # Relative position of star2 from COM

    star1_pos = rel_star1 @ R_inc.T + COM_vec  # Rotated relative position of star1
    star2_pos = rel_star2 @ R_inc.T + COM_vec  # Rotated relative position of star2

    # Apply inclination Rodrigues' rotation formula to the star velocities
    star1_velocity = star1_velocity @ R_inc.T  # Rotated velocity of star1
    star2_velocity = star2_velocity @ R_inc.T  # Rotated velocity of star1


    # Apply longitude of ascending node using Rodrigues' rotation formula
    # Get new relative position and velocities
    r_rel = star2_pos - star1_pos
    v_rel = star2_velocity - star1_velocity

    # Calculate the new specific angular momentum vector
    h_vec = np.cross(r_rel, v_rel)
    h_unit = h_vec / np.linalg.norm(h_vec)  # Normalize the specific angular momentum vector

    # Find the current longitude of ascending node
    n = np.cross([0, 0, 1], h_unit)
    if np.linalg.norm(n) < 1e-12:
        current_longitude_of_ascending_node = 0
        longitude_of_ascending_node = 0
    else:
        current_longitude_of_ascending_node = np.arctan2(n[1], n[0])

    # Calculate the Rodrigues' rotation matrix for longitude of ascending node
    R_long = rotate_about_axis([0, 0, 1], longitude_of_ascending_node - current_longitude_of_ascending_node)  # Rotate about z-axis of the COM of the binary system


    # Apply longitude of ascending node Rodrigues' rotation formula to the star position
    rel_star1 = star1_pos - COM_vec  # Relative position of star1 from COM
    rel_star2 = star2_pos - COM_vec  # Relative position of star2 from COM

    star1_pos = rel_star1 @ R_long.T + COM_vec # Rotated relative position of star1
    star2_pos = rel_star2 @ R_long.T + COM_vec # Rotated relative position of star2

    # Apply longitude of ascending node Rodrigues' rotation formula to the star velocities
    star1_velocity = star1_velocity @ R_long.T # Rotated velocity of star1
    star2_velocity = star2_velocity @ R_long.T # Rotated velocity of star1


    # Apply random argument of periapsis using Rodrigues' rotation formula
    # Calculate the eccentricity vector
    r_rel = star2_pos - star1_pos
    v_rel = star2_velocity - star1_velocity

    # Calculate the specific angular momentum vector
    h_vec = np.cross(r_rel, v_rel)
    h_unit = h_vec / np.linalg.norm(h_vec)

    # Calculate the longitude of ascending node vector
    longitude_of_ascending_node_vector = np.cross([0, 0, 1], h_unit)

    # Calculate the eccentricity vector
    mu = G * total_mass # Standard gravitational parameter
    eccentricity_vector = (np.cross(v_rel, h_vec) / mu) - (r_rel/ np.linalg.norm(r_rel))

    # Calculate the argument of periapsis
    norm_e = np.linalg.norm(eccentricity_vector)
    norm_long = np.linalg.norm(longitude_of_ascending_node_vector)

    # Special cases for finding the current argument of periapsis
    if norm_e < 1e-12 and norm_long <1e-12:
        current_argument_of_periapsis = 0.0 
    elif norm_e < 1e-12:
        current_argument_of_periapsis = 0.0
    elif norm_long < 1e-12:
        cosine = np.dot(eccentricity_vector, (1,0,0)) / (norm_e) # Reference direction is positive x-axis
        cosine = np.clip(cosine, -1.0, 1.0) # Clamp cosine-argument to [-1 , 1]
        current_argument_of_periapsis = np.arccos(cosine)

        # sin() disambiguation
        sin_argp = np.dot(np.cross((1, 0, 0), eccentricity_vector), h_unit)
        # Flip angle if sin is negative
        if sin_argp < 0:
            current_argument_of_periapsis = 2 * np.pi - current_argument_of_periapsis
    else:
        cosine = np.dot(eccentricity_vector, longitude_of_ascending_node_vector) / (norm_e * norm_long)
        cosine = np.clip(cosine, -1.0, 1.0) # Clamp cosine-argument to [-1, 1]
        current_argument_of_periapsis = np.arccos(cosine)

        # sin() disambiguation
        sin_argp = np.dot(np.cross(longitude_of_ascending_node_vector, eccentricity_vector), h_unit)
        # Flip angle if sin is negative
        if sin_argp < 0:
            current_argument_of_periapsis = 2 * np.pi - current_argument_of_periapsis

    # Calculate the argument of periapsis Rodrigues' rotation matrix 
    R_arg = rotate_about_axis(h_unit, argument_of_periapsis - current_argument_of_periapsis) # Rotational matrix about the normal axis of the orbital plane

    # Apply Rodrigues' rotation formula to the star position with random argument of periapsis
    rel_star1 = star1_pos - COM_vec  # Relative position of star1 from COM
    rel_star2 = star2_pos - COM_vec  # Relative position of star2 from COM

    star1_pos = rel_star1 @ R_arg.T + COM_vec # Rotated relative position of star1
    star2_pos = rel_star2 @ R_arg.T + COM_vec # Rotated relative position of star2

    # Apply argument of periapsis Rodrigues' rotation formula to the star velocities
    star1_velocity = star1_velocity @ R_arg.T # Rotated velocity of star1
    star2_velocity = star2_velocity @ R_arg.T # Rotated velocity of star2

    # Apply all transformation to the original full dataframe for comparison
    R_total = R_arg @ R_long @ R_inc # When transposed, it becomes R_inc first, then R_long and finally R_arg

    rel_full_star1 = np.stack([
        df['star1_x'] - df['COMx'],
        df['star1_y'] - df['COMy'],
        df['star1_z'] - df['COMz']], axis = 1)  # Relative position of star1 from COM (Shape: (N, 3))

    rel_full_star2 = np.stack([
        df['star2_x'] - df['COMx'],
        df['star2_y'] - df['COMy'],
        df['star2_z'] - df['COMz']], axis = 1)  # Relative position of star2 from COM (Shape: (N, 3))

    rotated_rel_star1 = rel_full_star1 @ R_total.T # Rotated relative position of star1 (Shape: (3, N))
    rotated_rel_star2 = rel_full_star2 @ R_total.T # Rotated relative position of star2 (Shape: (3, N))

    df['star1_x'] = rotated_rel_star1[:, 0] + df['COMx']
    df['star1_y'] = rotated_rel_star1[:, 1] + df['COMy']
    df['star1_z'] = rotated_rel_star1[:, 2] + df['COMz']
    df['star2_x'] = rotated_rel_star2[:, 0] + df['COMx']
    df['star2_y'] = rotated_rel_star2[:, 1] + df['COMy']
    df['star2_z'] = rotated_rel_star2[:, 2] + df['COMz']

    # Apply longitude of ascending node Rodrigues' rotation formula to the star velocities
    vel_star1 = np.stack([
        df['star1_vx'],
        df['star1_vy'],
        df['star1_vz']
    ], axis=1)

    vel_star2 = np.stack([
        df['star2_vx'],
        df['star2_vy'],
        df['star2_vz']
    ], axis=1)

    rotated_vel_star1 = vel_star1 @ R_total.T
    rotated_vel_star2 = vel_star2 @ R_total.T

    df['star1_vx'] = rotated_vel_star1[:, 0]
    df['star1_vy'] = rotated_vel_star1[:, 1]
    df['star1_vz'] = rotated_vel_star1[:, 2]
    df['star2_vx'] = rotated_vel_star2[:, 0]
    df['star2_vy'] = rotated_vel_star2[:, 1]
    df['star2_vz'] = rotated_vel_star2[:, 2]

    # Rebound setup and verification
    sim = rebound.Simulation()

    # Set gravitational constant based on unit system
    sim.G = G

    # Set units to SI units
    sim.units = units

    if scenarios_config.variations[base_variation_name].proper_motion_magnitude is not None:
        # Add proper motion velocity component
        if proper_motion_magnitude > 0:
            if proper_motion_direction is None:
                proper_motion_dir = np.array([1.0, 1.0, 0.0])
            else:
                proper_motion_dir = np.array(proper_motion_direction, dtype=float)
            proper_motion_dir /= np.linalg.norm(proper_motion_dir)
            proper_motion_velocity = proper_motion_dir * proper_motion_magnitude
            star1_velocity += proper_motion_velocity
            star2_velocity += proper_motion_velocity
            df[['star1_vx', 'star1_vy', 'star1_vz']] += proper_motion_velocity
            df[['star2_vx', 'star2_vy', 'star2_vz']] += proper_motion_velocity

    # Configure drag forces if specified
    drag_tau = scenarios_config.variations[base_variation_name].drag_tau
    if drag_tau is not None:
        sim.integrator = "ias15"  # Switch to adaptive integrator for non-conservative forces
        def apply_linear_drag(sim, particles, N=2):
            """Apply velocity-dependent linear drag force"""
            for i in range(N):
                particles[i].ax -= particles[i].vx / drag_tau
                particles[i].ay -= particles[i].vy / drag_tau
                particles[i].az -= particles[i].vz / drag_tau
        sim.additional_forces = lambda reb_sim: apply_linear_drag(reb_sim, sim.particles)
        sim.force_is_velocity_dependent = 1  # Required for velocity-dependent forces

    # Configure modified gravity if specified
    mod_gravity_exponent = scenarios_config.variations[base_variation_name].mod_gravity_exponent
    if mod_gravity_exponent is not None:
        sim.integrator = "ias15"  # Adaptive integrator for non-Newtonian forces
        def mod_gravity(reb, particles, N, mod_gravity_exponent):
            """Custom force implementation for modified gravity"""
            for i in range(N):
                for j in range(i+1, N):
                    # Calculate separation vector
                    dx = particles[j].x - particles[i].x
                    dy = particles[j].y - particles[i].y
                    dz = particles[j].z - particles[i].z
                    r = np.sqrt(dx**2 + dy**2 + dz**2)
                    # Modified gravity force: F ∝ 1/r^mod_gravity_exponent
                    F = sim.G * particles[i].m * particles[j].m / r**mod_gravity_exponent
                    # Apply forces to both particles
                    particles[i].ax = F * dx / (particles[i].m * r)
                    particles[i].ay = F * dy / (particles[i].m * r)
                    particles[i].az = F * dz / (particles[i].m * r)
                    particles[j].ax = -F * dx / (particles[j].m * r)
                    particles[j].ay = -F * dy / (particles[j].m * r)
                    particles[j].az = -F * dz / (particles[j].m * r)
        sim.additional_forces = lambda reb: mod_gravity(reb, sim.particles, N=2, mod_gravity_exponent=mod_gravity_exponent)


    # Add stars with initial conditions from the new tranformed DataFrame
    sim.add(m=m1, x=star1_pos[0], y=star1_pos[1], z=star1_pos[2], 
            vx=star1_velocity[0], vy=star1_velocity[1], vz=star1_velocity[2])
    sim.add(m=m2, x=star2_pos[0], y=star2_pos[1], z=star2_pos[2], 
            vx=star2_velocity[0], vy=star2_velocity[1], vz=star2_velocity[2])

    # Record the simulation data
    rows = []
    tvals = np.asarray(df['time'].values, dtype=np.float64)
    sim.t = 0

    for t in tvals:
        sim.integrate(t)  # Integrate the simulation to the current time
        p1 = sim.particles[0]
        p2 = sim.particles[1]

        # Calculate detailed orbital parameters
        separation = np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)
        force = sim.G * p1.m * p2.m / separation**2  # Newtonian force
        star1_accel = force / p1.m
        star2_accel = force / p2.m
        orbit = p2.orbit(primary=p1)  # Calculate orbital elements    

        detailed_row = [
                    t,
                    p1.x, p1.y, p1.z, p2.x, p2.y, p2.z,
                    p1.vx, p1.vy, p1.vz, p2.vx, p2.vy, p2.vz,
                    p1.m, p2.m, separation, force, star1_accel, star2_accel,
                    orbit.h, orbit.P, orbit.n, orbit.a, orbit.e,
                    orbit.inc, orbit.Omega, orbit.omega, orbit.f, orbit.M, orbit.T, orbit.d
                ]
        rows.append(detailed_row)

    # Convert to a pandas series
    sim_df = pd.DataFrame(rows, columns=[
                'time', 'star1_x', 'star1_y', 'star1_z', 'star2_x', 'star2_y', 'star2_z',
                'star1_vx', 'star1_vy', 'star1_vz', 'star2_vx', 'star2_vy', 'star2_vz', 
                'star1_mass', 'star2_mass', 'separation', 'force', 'star1_accel', 'star2_accel',
                'specific_angular_momentum', 'orbital_period', 'mean_motion', 'semimajor_axis', 
                'eccentricity', 'inclination', 'longitude_of_ascending_node', 'argument_of_periapsis','true_anomaly', 'mean_anomaly', 
                'time_of_pericenter_passage', 'radial_distance_from_reference'
            ])

    # Write to new files
    csv_file_detailed_sims = f"scenarios/detailed_sims/{base_variation_name}; Inc_{inclination:.2f}_Long_{longitude_of_ascending_node:.2f}_Arg_{argument_of_periapsis:.2f}; Trans_[{translation_x:.2g}, {translation_y:.2g}, {translation_z:.2g}].csv"
    with open(csv_file_detailed_sims, mode='w', newline='') as file_detailed_actual:
        sim_df.to_csv(file_detailed_actual, index=False)
        
    csv_file_sims = f"scenarios/sims/{base_variation_name}; Inc_{inclination:.2f}_Long_{longitude_of_ascending_node:.2f}_Arg_{argument_of_periapsis:.2f}; Trans_[{translation_x:.2g}, {translation_y:.2g}, {translation_z:.2g}].csv"
    with open(csv_file_sims, mode='w', newline='') as file_sims:
        sim_df[['time', 'star1_x', 'star1_y', 'star1_z', 'star2_x', 'star2_y', 'star2_z']].to_csv(file_sims, index=False)

    cols = ['star1_x', 'star1_y', 'star1_z', 'star2_x', 'star2_y', 'star2_z']
    print(f"Expected inc: {inclination}, Actual inc: {sim_df['inclination'].iloc[0]} \nExpected long: {longitude_of_ascending_node}, Actual long: {sim_df['longitude_of_ascending_node'].iloc[0]}, \nExpected argument_of_periapsis: {argument_of_periapsis}, Actual argument_of_periapsis: {sim_df['argument_of_periapsis'].iloc[0]}")
    for i in np.linspace(0, len(df)-1, num=1000, dtype=int):
        df_row = df.loc[i, cols].to_numpy()
        test_row = sim_df.loc[i, cols].to_numpy()
        assert np.allclose(df_row, test_row, rtol=1e-5), (
            f"Row {i} differs by more than rtol=1e-5:\n"
            f"abs diff = {np.abs(df_row - test_row)} \n"
            f"{df_row}, {test_row}, {csv_file_sims} \n"
            f"Expected inc: {inclination}, Actual inc: {sim_df['inclination'].iloc[0]} \nExpected long: {longitude_of_ascending_node}, Actual long: {sim_df['longitude_of_ascending_node'].iloc[0]}, \nExpected argument_of_periapsis: {argument_of_periapsis}, Actual argument_of_periapsis: {sim_df['argument_of_periapsis'].iloc[0]}")


def rotate_about_axis(axis, theta):
    """
    Rotate 3D vectors around an arbitrary axis.

    Parameters:
        axis : list or array of 3 floats
            Rotation axis direction (does not need to be unit length).
        theta : float
            Rotation angle in radians (positive = counterclockwise around axis).

    Returns:
        rotated_vectors : list of [x, y, z]
            Rotated vectors.
    """
    axis = np.array(axis, dtype=float)

    # Normalize the axis vector
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    one_minus_cos = 1 - cos_t

    # Rodrigues' rotation matrix
    R = np.array([
        [cos_t + x*x*one_minus_cos,
         x*y*one_minus_cos - z*sin_t,
         x*z*one_minus_cos + y*sin_t],

        [y*x*one_minus_cos + z*sin_t,
         cos_t + y*y*one_minus_cos,
         y*z*one_minus_cos - x*sin_t],

        [z*x*one_minus_cos - y*sin_t,
         z*y*one_minus_cos + x*sin_t,
         cos_t + z*z*one_minus_cos]
    ])

    return R

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))