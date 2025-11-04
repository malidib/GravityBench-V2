import numpy as np
import pandas as pd
import rebound
import os
import re
import ast
import ast
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scripts.scenarios_config as scenarios_config


def geometry(file_name:str, random=False, translation=False, verification=False):
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

    # Check file name in correct format
    filename_checking(file_name=file_name)

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
    star1_pos = np.array(scenarios_config.variations[base_variation_name].star1_pos)
    star2_pos = np.array(scenarios_config.variations[base_variation_name].star2_pos)

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

    if verification == True:
        cols = ['star1_x', 'star1_y', 'star1_z', 'star2_x', 'star2_y', 'star2_z']
        print(f"Expected inc: {inclination}, Actual inc: {sim_df['inclination'].iloc[0]} \nExpected long: {longitude_of_ascending_node}, Actual long: {sim_df['longitude_of_ascending_node'].iloc[0]}, \nExpected argument_of_periapsis: {argument_of_periapsis}, Actual argument_of_periapsis: {sim_df['argument_of_periapsis'].iloc[0]}")
        for i in np.linspace(0, len(df)-1, num=1000, dtype=int):
            df_row = df.loc[i, cols].to_numpy()
            test_row = sim_df.loc[i, cols].to_numpy()
            assert np.allclose(df_row, test_row, rtol=1e-6), (
                f"Row {i} differs by more than rtol=1e-6:\n"
                f"abs diff = {np.abs(df_row - test_row)} \n"
                f"{df_row}, {test_row}, {csv_file_sims} \n"
                f"Expected inc: {inclination}, Actual inc: {sim_df['inclination'].iloc[0]} \nExpected long: {longitude_of_ascending_node}, Actual long: {sim_df['longitude_of_ascending_node'].iloc[0]}, \nExpected argument_of_periapsis: {argument_of_periapsis}, Actual argument_of_periapsis: {sim_df['argument_of_periapsis'].iloc[0]}")

            
    return f"{base_variation_name}; Inc_{inclination:.2f}_Long_{longitude_of_ascending_node:.2f}_Arg_{argument_of_periapsis:.2f}; Trans_[{translation_x:.2g}, {translation_y:.2g}, {translation_z:.2g}]"


# Strict float
_FLOAT = r"[+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?"

pattern = re.compile(fr"""
    ^\s*
    (?P<base>[^;]+?)                       # everything before first ';'
    (?:                                    # optional geometry block
      ;\s+                                 # ← semicolon + at least one space
      (?:
        Inc_(?P<inc>{_FLOAT})
        (?:
           _Long_(?P<long>{_FLOAT})
           (?:
             _Arg_(?P<arg>{_FLOAT})
           )?
        |
           _Arg_(?P<arg_inc_only>{_FLOAT})
        )?
      |
        Long_(?P<long_only>{_FLOAT})
        (?:
          _Arg_(?P<arg_long_only>{_FLOAT})
        )?
      |
        Arg_(?P<arg_only>{_FLOAT})
      )
    )?
    (?:                                    # optional translation block
      ;\s+                                 # ← semicolon + at least one space
      Trans_\[
        (?P<tx>{_FLOAT})\s*,\s*(?P<ty>{_FLOAT})\s*,\s*(?P<tz>{_FLOAT})
      \]
    )?
    \s*$
""", re.VERBOSE)

def filename_checking(file_name: str):
    """
    Check whether filename is in correct format.

    Expect file_name of the form:
      "<base>; Inc_<inc>_Long_<long>_Arg_<arg>; Trans_[<tx>, <ty>, <tz>]"

    It also accepts any specific arguments, for example "Inc_0.2_Arg_1.4" which will only orient inclination and argument of periapsis, leaving others unchanged. But it must be in the sequence above.

    Parameters:
    ----------
    file_name: str
        Name of the file being passed through geometry() function

    Return:
    ------
    Does not return anything, but raises ValueError if it doesn’t match exactly.
    """

    if not pattern.match(file_name):
        raise ValueError(f"Filename not in expected format: \"<base>; Inc_<inc>_Long_<long>_Arg_<arg>; Trans_[<tx>, <ty>, <tz>]\". Your filename: {file_name}")


# Projection, the observer is at the origin, and change coordinates from xyz to x'y' for projection
def projection(obs_df, file_name: str, save=False):
    """
    Projection of binary system from a xyz coordinate onto x'y' a new plane. It works through the following

    1. Find the vector to the starting COM of the binary system, we will use this vector as the normal vector to the projected plane,
    centred at the origin.
    2. Perform orthogonal projection from the xyz binary system onto this plane.
    3. Then find the right ascension and declination using the COM vector.
    4. Rotate the projected plane so that the normal vector of the plane aligns above or below the positive y-axis 
    5. Rotate the projected plane so that the normal vector of the plane aligns directly along the negative z-axis

    Note: Test results
    It gets close to zero, where star1 and star2 z components are about 10^-5.  

    In the case of projecting an orbit fully face-off, the y-axis becomes much smaller, in magnitudes of 10^-20.

    Parameters:
    -----------
    obs_df : pandas.DataFrame
        DataFrame that we want to project
        A dataframe is called so that it can be used in row_wise cases
    file_name : str
        Name of the original variation
    save: bool
        Whether to writes to a new csv file in scenarios/projected_sims
        It is named as follows "{file_name}.csv"

    Returns:
    --------
    pandas.DataFrame
        New dataframe that has been projected
    """

    # Read csv
    df = pd.read_csv(f"scenarios/detailed_sims/{file_name}.csv")

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

    star1_coord = np.vstack([obs_df['star1_x'], obs_df['star1_y'], obs_df['star1_z']])  # (3, N)
    star2_coord = np.vstack([obs_df['star2_x'], obs_df['star2_y'], obs_df['star2_z']])

    # Project “centered” positions onto the plane by removing the normal component: v_proj = v - (n·v) n
    proj_norm1 = (COM_unit_vec @ star1_coord) * COM_unit_vec[:, None]  # (3, N)
    proj_norm2 = (COM_unit_vec @ star2_coord) * COM_unit_vec[:, None]

    star1_pro = star1_coord - proj_norm1
    star2_pro = star2_coord - proj_norm2

    # Update the dataframe to contain our new position data
    obs_df['star1_x'], obs_df['star1_y'], obs_df['star1_z'] = star1_pro
    obs_df['star2_x'], obs_df['star2_y'], obs_df['star2_z'] = star2_pro

    z = np.array([0.0, 0.0, 1.0])

    # If nhat ~ +/- z, fallback to y as 'north' source to avoid degeneracy
    # This is the case when the los points directly upwards along z, then take 
    # positive y as north

    if abs(np.dot(z, COM_unit_vec)) > 1.0 - 1e-10:
        y = np.array([0.0, 1.0, 0.0])
        north = unit(y - np.dot(y, COM_unit_vec) * COM_unit_vec)
    else:
        north = unit(z - np.dot(z, COM_unit_vec) * COM_unit_vec)
    east = unit(np.cross(COM_unit_vec, north))

    pos_star1 = np.stack([
        obs_df['star1_x'],
        obs_df['star1_y'],
        obs_df['star1_z']], axis = 1)  # Relative position of star1 from COM (Shape: (N, 3))
    
    pos_star2 = np.stack([
        obs_df['star2_x'],
        obs_df['star2_y'],
        obs_df['star2_z']], axis = 1)  # Relative position of star2 from COM (Shape: (N, 3))
    
    x1prime = pos_star1 @ east
    x2prime = pos_star2 @ east
    y1prime = pos_star1 @ north
    y2prime = pos_star2 @ north

    obs_df['star1_x'] = x1prime
    obs_df['star1_y'] = y1prime
    obs_df['star1_z'] = 0.0
    obs_df['star2_x'] = x2prime
    obs_df['star2_y'] = y2prime
    obs_df['star2_z'] = 0.0

    # Calculate the right ascension
    dot = np.dot([COMx, COMy, 0], [1, 0, 0])
    na = np.linalg.norm([COMx, COMy, 0])
    nb = np.linalg.norm([1, 0, 0])
    
    right_ascension = np.arccos(dot/(na * nb))

    # Check for sign ambiguity, np.arccos is between [0, pi]. We want a counterclockwise angle from [0, 2*pi] from positive x-axis
    if COMy < 0:
        right_ascension = 2*np.pi - right_ascension

    # Calculate the declination, np.arctan2 returns [-pi, pi]
    dec = np.arcsin(COM_unit_vec[2])

    # If want to save to a new file, write to a new folder 
    if save == True:
        obs_df['right_ascension'] = right_ascension
        obs_df['declination'] = dec
        csv_file_sims = f"scenarios/projected_sims/{file_name}.csv"
        with open(csv_file_sims, mode='w', newline='') as file_sims:
            obs_df[['time', 'star1_x', 'star1_y', 'star1_z', 'star2_x', 'star2_y', 'star2_z', 'right_ascension', 'declination']].to_csv(file_sims, index=False)

    return obs_df


# Function to check for valid star positions for projection
def pre_projection_checking(df, file_name:str, tol=None, align_tol=1e-10):
    """
    Check whether the df containing xyz star positions are valid for projection. There are two cases to check for:
    1. The stars does not pass through origin (colliding with earth)
    2. The separation vector does not cut through the origin, or else projection will not make sense.

    To do the second, we use separation vector, and we can use cross and dot product to check whether the separation vector cuts the origin, r1 and r2 collinear
    and that the positive vector r1 and r2 are in opposite direction.

    Parameters:
    ----------
    df: pandas.DataFrame
        Dataframe containing xyz star positions
    file_name: str
        Name of the file
    tol: float
        Tolerance for checking norms, if None, it will be calculated based on the median scale
    align_tol: float
        Tolerance for checking alignment of vectors, default is 1e-10
    
    Returns:
    ------
    ValueError it fails the two cases mentioned
    """

    # Get star positions
    r1 = df[['star1_x', 'star1_y', 'star1_z']].to_numpy(dtype=float)  # (N,3)
    r2 = df[['star2_x', 'star2_y', 'star2_z']].to_numpy(dtype=float)

    # Scale-aware tolerance if not given
    if tol is None:
        typical_scale = np.median(np.linalg.norm(np.vstack([r1, r2]), axis=1)) or 1.0
        tol = max(typical_scale * 1e-12, 1e-10)  # distance tolerance

    # Norms
    n1 = np.linalg.norm(r1, axis=1)
    n2 = np.linalg.norm(r2, axis=1)

    # Check if star passes the origin, we can do this with norm, and checking if it goes to zero. 
    star1_hit_idx = np.where(n1 < tol)[0] # Indices of r1 that has norm goes to 0
    if star1_hit_idx.size:
        i = int(star1_hit_idx[0])
        raise ValueError(
            f"Variation {file_name}: star 1 passes the origin at row {i}. Collision invalid for projection. Translate or change parameters."
        )

    star2_hit_idx = np.where(n2 < tol)[0] # Indices of r1 that has norm goes to 0
    if star2_hit_idx.size:
        i = int(star2_hit_idx[0])
        raise ValueError(
            f"Variation {file_name}: star 2 passes the origin at row {i}. Collision invalid for projection. Translate or change parameters."
        )

    # Check for whether the separation vector of the two stars passes the origin, if so projection becomes invalid, it physically does not make sense.
    # This happens when r1 and r2 are collinear
    cross_product_values = np.linalg.norm(np.cross(r1, r2), axis=1) # 0 iff collinear
    dot_product_values = np.einsum('ij,ij->i', r1, r2) # < 0 means that r1 and r2 are in opposite direction

    collinear = (cross_product_values <= align_tol * n1 * n2)
    opposite  = (dot_product_values < 0.0)
    good_norm =  (n1 >= tol) & (n2 >= tol) # exclude zero vectors (handled as collisions)

    cuts_origin_idx = np.where(collinear & opposite & good_norm)[0] # Indices of the separation vector cutting the origin.
    if cuts_origin_idx.size:
        i = int(cuts_origin_idx[0])
        raise ValueError(
            f"Projection invalid for variation {file_name} at row {i}. The separation vector passes through the origin, leading to invalid projections. Translate or change parameters."
        )



# Remove the randomly transformed binary orbit csv files. This function deletes a json file with random geometry and renames the original base variation json file. 
# This is essentially a reset.
def reset():
    folder_path_detailed = "scenarios/detailed_sims"
    file_names = os.listdir(folder_path_detailed) # Same name for both sims and detailed_sims
    projected_file_names = os.listdir("scenarios/projected_sims") # Projected sims files

    # Remove all projected_sims csv files
    for file in projected_file_names:
        file_path_projected_sims = f"scenarios/projected_sims/{file}"
        if os.path.exists(file_path_projected_sims):
            os.remove(file_path_projected_sims)

    # Remove all transformed binary orbit csv files
    for file in file_names:
        if "Inc" in file:
            file_path_detailed = f"scenarios/detailed_sims/{file}"
            file_path_sims = f"scenarios/sims/{file}"
            if os.path.exists(file_path_detailed):
                os.remove(file_path_detailed)
            if os.path.exists(file_path_sims):
                os.remove(file_path_sims)



# Sky projection, xyz coordinates onto new x'y'z' coordinates that's in a plane projected on the sky. The observer is at the origin.
def sky_projection(df, file_name: str, save=False):
    """
    Projection of xyz orbit onto a plane normal to the view from origin, centered at the initial COM of the system. This projection provides
    a new xyz coordinates, but in a projected plane in xyz coordinates

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing simulation data with position and time columns (detailed_sims)
    file_name : str
        Name of the original variation

    Returns:
    --------
    Writes to a new csv file in scenarios/sims
        It is named as follows {file_name}; sky_projected
    """

    # Ensure we don't modify the original variation
    df = df.copy(deep=True)

    # Find the center of mass of the binary system
    # Get masses for COM calculation
    m1, m2 = df['star1_mass'].iloc[0], df['star2_mass'].iloc[0]
    total_mass = m1 + m2

    # Calculate COM coordinates
    df['COMx'] = (m1*df['star1_x'] + m2*df['star2_x'])/total_mass
    df['COMy'] = (m1*df['star1_y'] + m2*df['star2_y'])/total_mass
    df['COMz'] = (m1*df['star1_z'] + m2*df['star2_z'])/total_mass

    # Take the initial centre of mass as the reference point for the centre of projection
    COMx = df['COMx'].iloc[0]
    COMy = df['COMy'].iloc[0]
    COMz = df['COMz'].iloc[0]

    COM_vec = np.array([COMx, COMy, COMz])
    norm = np.linalg.norm(COM_vec)
    COM_unit_vec = COM_vec / norm  # Normal unit vector to the plane

    star1_coord = np.vstack([df['star1_x'], df['star1_y'], df['star1_z']])  # (3, N)
    star2_coord = np.vstack([df['star2_x'], df['star2_y'], df['star2_z']])

    # Translate so initial COM is at the origin
    centered1 = star1_coord - COM_vec[:, None]   # (3, N)
    centered2 = star2_coord - COM_vec[:, None]

    # Project “centered” positions onto the plane by removing the normal component: v_proj = v - (n·v) n
    proj_norm1 = (COM_unit_vec @ centered1) * COM_unit_vec[:, None]  # (3, N)
    proj_norm2 = (COM_unit_vec @ centered2) * COM_unit_vec[:, None]

    star1_pro = centered1 - proj_norm1
    star2_pro = centered2 - proj_norm2

    df['star1_x'], df['star1_y'], df['star1_z'] = star1_pro + COM_vec[:, None]
    df['star2_x'], df['star2_y'], df['star2_z'] = star2_pro + COM_vec[:, None]

    # If want to save to a new file, write to a new folder 
    if save == True:
        csv_file_sims = f"scenarios/projected_sims/{file_name}.csv"
        with open(csv_file_sims, mode='w', newline='') as file_sims:
            df[['time', 'star1_x', 'star1_y', 'star1_z', 'star2_x', 'star2_y', 'star2_z']].to_csv(file_sims, index=False)


# Helper function to rotate vectors about an arbitrary axis using Rodrigues' rotation formula
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

def unit(v):
    """
    Return the unit vector of input. Checks for zero magnitude errors.
    """
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("Zero-length vector")
    return v / n


def main(reset_var=False):
    if reset_var:
        reset()
        print("INTERNAL: Variations removed.")
    else:
        print("No cleanup requested.")

if __name__ == "__main__":
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Clean up transformed variations')
    
    # Scenario configuration
    parser.add_argument('--reset', action='store_true', default=False,
                       help='Remove any transformed variation csv files, leaving the original base variation csv files intact.')

    args = parser.parse_args()

    main(reset_var=args.reset)