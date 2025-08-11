"""
The objective of this script is to include more realistic parameters such as luminosity and eclipsing binaries.

It also tried to infer x-direction through the use of different parameters.
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import json

# Calculate luminosity of the stars, under the assumption that the stars are on the main-sequence. Therefore, it is best to keep the 
# binary star masses between 2 to 55 solar masses.
def luminosity_generator(file_name:str, save=False, main_sequence=True):
    """
    Generate the luminosity of the stars based on their mass. The luminosity is calculated using the mass-luminosity 
    relation for main-sequence stars. 
    
    The relation is given by: L = Lsun * (M/Msun)**a
    where Lsun is the luminosity of the Sun, M is the mass of the star, Msun is the mass of the Sun, and a is a constant. 
    
    For main-sequence stars, a is approximately 3.5

    file_name: name of the file to read and write
    save: whether to save the luminosity columns to the dataframe csv or return the luminosity values

    Parameters:
    -----------
    file_name : str
        Name of the file containing the simulation data
    save : bool
        Whether to save the luminosity columns to the dataframe csv or return the luminosity values

    Returns:
    --------
    Lstar1, Lstar2 : float
        Luminosity of the first and second star
    """

    # Load the simulation data
    df = pd.read_csv(f"scenarios/detailed_sims/{file_name}.csv")

    # Setup parameters
    Msun = 1.9891e30
    Lsun = 3.8395e26
    a = 3.5
    
    Mstar1 = df['star1_mass'].iloc[0]
    Mstar2 = df['star2_mass'].iloc[0]
    Lstar1 = Lsun * ((Mstar1/Msun)**a)
    Lstar2 = Lsun * ((Mstar2/Msun)**a)

    if save:
        df['star1_lum'] = np.full(len(df), Lstar1)
        df['star2_lum'] = np.full(len(df), Lstar1)
        df.to_csv(f"scenarios/projected_sims/{file_name}.csv", index=False)
        return Lstar1, Lstar2
    else:
        return Lstar1, Lstar2


# Calculate the flux measured, the reference point is the origin
def calculate_flux(file_name:str, save=False):
    """
    Calculate the flux of each star at each time step based on their luminosity and distance from the origin.

    Parameters:
    -----------
    file_name : str
        Name of the file containing the simulation data
    save : bool
        Whether to save the flux columns to the dataframe csv or return the flux values

    Returns:
    --------
    Fstar1, Fstar2 : np arrays of float
        Flux of the first and second star at each time step
    """

    # Load the simulation data
    df = pd.read_csv(f"scenarios/projected_sims/{file_name}.csv")

    # Calculate luminosity of each star
    Lstar1, Lstar2 = luminosity_generator(file_name=file_name, save=False)
    r_star1 = np.sqrt(df['star1_x']**2 + df['star1_y']**2 + df['star1_z']**2)
    r_star2 = np.sqrt(df['star2_x']**2 + df['star2_y']**2 + df['star2_z']**2)
    Fstar1 =  Lstar1 / (4*np.pi*(r_star1**2))
    Fstar2 =  Lstar2 / (4*np.pi*(r_star2**2))

    if save:
        df['star1_flux'] = Fstar1
        df['star2_flux'] = Fstar2
        df.to_csv(f"scenarios/projected_sims/{file_name}.csv", index=False)
        return Fstar1, Fstar2
    else:
        return Fstar1, Fstar2

# Calculate the absolute magnitude of each star at each time
def absolute_magnitude(file_name:str, save=False):
    """
    Calculate the absolute magnitude of each star at each time and position.
    The absolute magnitude is calculated using the formula:
    m = Mag_sun - 2.5 * log10(Lstar/Lsun)
    where Mag_sun is the absolute magnitude of the Sun, Lstar is the luminosity of the star, and Lsun is the luminosity of the Sun.

    The absolute magnitude of the Sun is 4.74.

    Parameters:
    -----------
    file_name : str
        Name of the file containing the simulation data
    save : bool
        Whether to save the absolute magnitude columns to the dataframe csv or return the absolute magnitude values

    Returns:
    --------
    abs_mag_star1, abs_mag_star2 : float
        Absolute magnitude of the first and second star
    """

    # Load the simulation data
    df = pd.read_csv(f"scenarios/detailed_sims/{file_name}.csv")

    # Setup parameters
    Lsun = 3.839e26
    abs_mag_sun = 4.74
    Lstar1, Lstar2 = luminosity_generator(file_name=file_name, save=False)

    abs_mag_star1 = abs_mag_sun - 2.5*np.log10(Lstar1/Lsun)
    abs_mag_star2 = abs_mag_sun - 2.5*np.log10(Lstar2/Lsun)

    if save:
        df['star1_abs_mag'] = np.full(len(df), abs_mag_star1)
        df['star2_abs_mag'] = np.full(len(df), abs_mag_star2)
        df.to_csv(f"scenarios/projected_sims/{file_name}.csv", index=False)
        return abs_mag_star1, abs_mag_star2
    else:
        return abs_mag_star1, abs_mag_star2
    
# Calculate the apparent magnitude of each star at each time
def apparent_magnitude(file_name:str, save=False):
    """
    Calculate the apparent magnitude of each star at each time and position.
    The apparent magnitude is calculated using the formula:
    m = M + 5*log10(d/10pc)
    where M is the absolute magnitude of the star, d is the distance to the star from origin. This formula is derived from distance modulus formula

    The absolute magnitude of the Sun is 4.74.

    Parameters:
    -----------
    file_name : str
        Name of the file containing the simulation data
    save : bool
        Whether to save the apparent magnitude columns to the dataframe csv or return the apparent magnitude values

    Returns:
    --------
    app_mag_star1, app_mag_star2 : np arrays of float
        Apparent magnitude of the first and second star at each time step
    """

    # Load the simulation data
    df = pd.read_csv(f"scenarios/detailed_sims/{file_name}.csv")

    # Setup parameters
    abs_mag_star1, abs_mag_star2 = absolute_magnitude(file_name=file_name, save=False)
    abs_mag_star1 = np.full(len(df), abs_mag_star1)
    abs_mag_star2 = np.full(len(df), abs_mag_star2)

    pc_to_m = 3.0856776e16

    r_star1 = np.sqrt(df['star1_x']**2 + df['star1_y']**2 + df['star1_z']**2)
    r_star2 = np.sqrt(df['star2_x']**2 + df['star2_y']**2 + df['star2_z']**2)

    app_mag_star1 = abs_mag_star1 + 5*np.log10(r_star1/(10 * pc_to_m))
    app_mag_star2 = abs_mag_star2 + 5*np.log10(r_star2/(10 * pc_to_m))

    if save:
        df['star1_app_mag'] = app_mag_star1
        df['star2_app_mag'] = app_mag_star2
        df.to_csv(f"scenarios/projected_sims/{file_name}.csv", index=False)
        return app_mag_star1, app_mag_star2
    else:
        return app_mag_star1, app_mag_star2
    
    # Another method would be to do m = Msun - 2.5*log10(F/Fsun_10pc), where Msun is absolute magnitude of sun

# We could just add absolute and apparent magnitude, and let the agent figure the rest out.

# Infer star position
def infer_position(file_name:str, verification=False, save=False):
    """
    Infer the position of the stars in a binary system, given their absolute and apparent magnitudes. Assumes that the stars are on the main-sequence
    and that the projected plane works as described.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the detailed simulation data
    verification : bool
        Whether to verify values matches with the actual x positions
    save : bool
        Whether to return the empirically derived value or the value inputted into the simulation.
    
    Returns:
    --------
    pd.DataFrame
        Containing 'time', 'star1_x', 'star1_y', 'star1_z', 'star2_x', 'star2_y', 'star2_z'
    """

    # Check valid file
    if not os.path.exists(f"scenarios/projected_sims/{file_name}.csv"):
        raise FileNotFoundError(f"Projected {file_name} file not found, please check the file name is correct, or that this file was ran with projection=True.")

    # Load the simulation data
    df = pd.read_csv(f"scenarios/detailed_sims/{file_name}.csv")
    df_pro = pd.read_csv(f"scenarios/projected_sims/{file_name}.csv")

    dec, right_ascension = df_pro['declination'].iloc[0], df_pro['right_ascension'].iloc[0]

    # Find the Rodigues' rotation matrix to rotate to the correct declination
    
    R_dec = rotate_about_axis([1, 0, 0], np.pi/2 + dec)

    R_ra = rotate_about_axis([0, 0, 1], right_ascension - np.pi/2)

    R = R_ra @ R_dec

    # Apply the rotation matrix to the projected coordinates
    pos_star1 = np.stack([
        df_pro['star1_x'],
        df_pro['star1_y'],
        df_pro['star1_z']], axis = 1)  # Relative position of star1 from COM (Shape: (N, 3))
    
    pos_star2 = np.stack([
        df_pro['star2_x'],
        df_pro['star2_y'],
        df_pro['star2_z']], axis = 1)  # Relative position of star2 from COM (Shape: (N, 3))
    
    rotated_star1 = pos_star1 @ R.T # Rotated relative position of star1 (Shape: (N, 3))
    rotated_star2 = pos_star2 @ R.T # Rotated relative position of star2 (Shape: (N, 3))

    df_pro['star1_x'] = rotated_star1[:, 0]
    df_pro['star1_y'] = rotated_star1[:, 1]
    df_pro['star1_z'] = rotated_star1[:, 2]
    df_pro['star2_x'] = rotated_star2[:, 0]
    df_pro['star2_y'] = rotated_star2[:, 1]
    df_pro['star2_z'] = rotated_star2[:, 2]

    # Find COM xyz positions
    COMz = np.sin(dec)
    COMx = np.cos(right_ascension)
    COMy = np.sin(right_ascension)

    COM_vec = np.array([COMx, COMy, COMz])

    # Calculate the distance modulus for each star
    abs_mag_star1, abs_mag_star2 = absolute_magnitude(file_name=file_name, save=False)

    # Fill the value into a numpy array
    abs_mag_star1 = np.full(len(df), abs_mag_star1)
    abs_mag_star2 = np.full(len(df), abs_mag_star2)

    app_mag_star1, app_mag_star2 = apparent_magnitude(file_name=file_name, save=False)

    distance_modulus_star1 = app_mag_star1 - abs_mag_star1
    distance_modulus_star2 = app_mag_star2 - abs_mag_star2

    pc_to_m = 3.0856776e16

    distance_star1 = (10**((distance_modulus_star1 + 5)/5)) * pc_to_m
    distance_star2 = (10**((distance_modulus_star2 + 5)/5)) * pc_to_m

    # Unit vector of the projected sky plane
    n_unit = COM_vec/np.linalg.norm(COM_vec)

    # (-b + sqrt(b^2 -4ac))/ 2a
    # Plus sign is used here as we are assuming that the unit vector always points away from the origin in the direction of the projected plane, 
    # and that our projected plane is always closer to the origin than actual star positions.
    a = np.dot(n_unit, n_unit)
        
    # Extract positions as (N,3) arrays
    star1_pos = df_pro[['star1_x','star1_y','star1_z']].to_numpy()
    star2_pos = df_pro[['star2_x','star2_y','star2_z']].to_numpy()

    b_star1 = 2*(star1_pos @ n_unit)
    b_star2 = 2*(star2_pos @ n_unit)

    c_star1 = np.sum(star1_pos**2, axis=1) - distance_star1**2
    c_star2 = np.sum(star2_pos**2, axis=1) - distance_star2**2

    t_star1 = (-b_star1 + np.sqrt((b_star1**2) - 4*a*c_star1))/(2*a)
    t_star2 = (-b_star2 + np.sqrt((b_star2**2) - 4*a*c_star2))/(2*a)

    t_star1 = np.asarray(t_star1)
    t_star2 = np.asarray(t_star2)

    time = np.asarray(df_pro['time'])

    star1_pos += t_star1[:, None] * n_unit[None, :]
    star2_pos += t_star2[:, None] * n_unit[None, :]

    combined = np.hstack([time[:, None] , star1_pos, star2_pos])

    result_df = pd.DataFrame(combined, columns=['time', 'star1_x', 'star1_y', 'star1_z', 'star2_x', 'star2_y', 'star2_z'])

    # Verification step
    if verification:
        max_diff = 0
        threshold_value = 1000
        rows = []
        for i in range(len(df)):
            df_row = df.iloc[i]
            test_row =result_df.iloc[i]
            current_max_diff = max(abs(df_row['star1_x'] - test_row['star1_x']), 
                                   abs(df_row['star1_y'] - test_row['star1_y']),
                                   abs(df_row['star1_z'] - test_row['star1_z']), 
                                   abs(df_row['star2_x'] - test_row['star2_x']),
                                   abs(df_row['star2_y'] - test_row['star2_y']), 
                                   abs(df_row['star2_z'] - test_row['star2_z']),
                                   )
            rows.append(current_max_diff)
            if current_max_diff > max_diff:
                max_diff = current_max_diff
            assert max_diff <= threshold_value, f"Difference between points found to be more than {threshold_value}, at {i} row."
    
    if save:
        result_df.to_csv(f"scenarios/projected_sims/{file_name}_unprojected.csv", index=False)
        print(f"{max_diff:.2g}")

    return result_df


# Remove parameters used in this file columns
def remove(file_name:str):
    """
    Remove the luminosity, flux and magnitude columns of the file dataframe.

    Parameters:
    -----------
    file_name : str
        Name of the file containing the detailed simulation data
    """

    # Load the simulation data
    df = pd.read_csv(f"scenarios/detailed_sims/{file_name}.csv")

    # Check for columns and drop them if they exist
    cols = ['star1_lum', 'star2_lum', 'star1_flux', 'star2_flux', 'star1_abs_mag', 'star2_abs_mag', 'star1_app_mag', 'star2_app_mag',]
    cols_to_drop = [col for col in cols if col in df.columns]
    df = df.drop(columns=cols_to_drop)

    # Write the dataframe back to the csv file
    df.to_csv(f"scenarios/detailed_sims/{file_name}.csv", index=False)



# Calculate radius, we also assume main-sequence stars with an estimation of R = M**0.8, where R is in solar radii and M is in solar masses.
def mass_to_radius(file_name:str, add=False):
    """
    Calculate radius of the star through a main sequence mass-radius relation

    Formula: R = M ** 0.8, where M is in solar masses and R is in solar radii

    Parameters:
    -----------
    file_name : str
        Name of the file containing the simulation data
    add : bool
        Whether to add the radius of the stars to the original simulation data or not

    Returns:
    --------
    radius_star1, radius_star2 : float
        Radius of the stars
    """

    Msun = 1.989e30
    Rsun = 6.957e8

    df = pd.read_csv(f"scenarios/detailed_sims/{file_name}.csv")
    mass_star1 = df['star1_mass'].iloc[0]
    mass_star2 = df['star2_mass'].iloc[0]
    b = 0.8
    radius_star1 = ((mass_star1/Msun)**b)* Rsun # In meters
    radius_star2 = ((mass_star2/Msun)**b)* Rsun # In meters

    if add:
        df['star1_radius'] = np.full(len(df), radius_star1)
        df['star2_radius'] = np.full(len(df), radius_star2)
        df.to_csv(f"scenarios/detailed_sims/{file_name}.csv", index=False)
    else:
        return radius_star1, radius_star2

# We remove datas when stars pass by each other. Different levels of eclipsing binaries can be simulated, where 2 levels are chosen for different accuracy.
def eclipsing_binary(file_name:str, accuracy=0):
    """
    Remove data points where stars pass by each other, based on the accuracy level.

    accuracy: 0 - keep all data points (no eclipsing binary)
              1 - remove the data points when a star partially and fully covers another star (considered partially eclipsing binary)
              2 - remove data points only when a star fully covers another star (considered fully eclipsing binary)
    file_name: name of the file to read and write  
    """

    if accuracy == 0:
        return

    # Setup the boundary points
    radius_star1, radius_star2 = mass_to_radius(file_name, add=False)
    eb_setup(file_name=file_name, radius_star1=radius_star1, radius_star2=radius_star2, add=True)

    df = pd.read_csv(f"scenarios/detailed_sims/{file_name}.csv")
    df = df.copy(deep=True)

    df['star1_boundary'] = df['star1_boundary'].apply(lambda s: np.array(json.loads(s)))
    df['star2_boundary'] = df['star2_boundary'].apply(lambda s: np.array(json.loads(s)))

    # Find COM as reference plane for projection

    # Find the center of mass of the binary system
    # Get masses for COM calculation
    m1, m2 = df['star1_mass'].iloc[0], df['star2_mass'].iloc[0]
    total_mass = m1 + m2

    # Calculate COM coordinates
    df['COMx'] = (m1*df['star1_x'] + m2*df['star2_x'])/total_mass
    df['COMy'] = (m1*df['star1_y'] + m2*df['star2_y'])/total_mass
    df['COMz'] = (m1*df['star1_z'] + m2*df['star2_z'])/total_mass

    COMx = df['COMx'].mean()
    COMy = df['COMy'].mean()
    COMz = df['COMz'].mean()

    COM = np.array([COMx, COMy, COMz])
    norm = np.linalg.norm(COM)
    COM_unit_vec = COM / norm  # Normal unit vector to the plane

    ref_coord = np.array([COMx, COMy, COMz])

    eclipse_scores = []
    primary_star_list = []

    # Project each circumeference point onto a plane
    for i in tqdm(range(len(df)), desc="Running calculations with boundary points"):
        row = df.iloc[i]
        star1_boundary_points = row['star1_boundary']
        star2_boundary_points = row['star2_boundary']

        star1_rel = np.sqrt(row['star1_x']**2 + row['star1_y']**2 + row['star1_z']**2)
        star2_rel = np.sqrt(row['star2_x']**2 + row['star2_y']**2 + row['star2_z']**2)

        # Check which star is in front, and let that be primary star
        if star1_rel > star2_rel:
            primary_star = star2_boundary_points
            secondary_star = star1_boundary_points
            primary_no = 2 # primary_no keeps track of with star is the primary
        else:
            primary_star = star1_boundary_points
            secondary_star = star2_boundary_points
            primary_no = 1
        
        primary_projected_coords = []
    
        for j in primary_star:
            proj_onto_normal = j - ((np.dot(j - ref_coord, COM_unit_vec)) * COM_unit_vec)
            primary_projected_coords.append(proj_onto_normal)

        primary_projected_coords = np.array(primary_projected_coords)

        N = len(primary_projected_coords)
    
        z_upper_limit = primary_projected_coords[0][2]
        z_lower_limit = primary_projected_coords[(N//2)-1][2] # Assumes that there will be one on top and one directly below

        y_upper_limit = primary_projected_coords[N//4][1]
        y_lower_limit = primary_projected_coords[3 * N // 4][1]
        
        current_eclipse_score=0

        for k in secondary_star:
            proj_onto_normal = k - ((np.dot(k - ref_coord, COM_unit_vec)) * COM_unit_vec)
            if (z_lower_limit <= proj_onto_normal[2] <= z_upper_limit) and (y_lower_limit <= proj_onto_normal[1] <= y_upper_limit):
                current_eclipse_score += 1
            
        eclipse_scores.append(current_eclipse_score)
        primary_star_list.append(primary_no)


    df['eclipse_score'] = np.array(eclipse_scores)
    df['primary_star'] = np.array(primary_star_list)

    # Convert each (n,3) numpy array to a JSON string with commas
    df['star1_boundary'] = df['star1_boundary'].apply(lambda arr: json.dumps(arr.tolist()))
    df['star2_boundary'] = df['star2_boundary'].apply(lambda arr: json.dumps(arr.tolist()))
    df.to_csv(f"scenarios/detailed_sims/{file_name}.csv", index=False)

    if accuracy == 1:
        eb_acc_1(df)
    else:
        eb_acc_2(df)

    csv_file_sims = f"scenarios/sims/{file_name}_EB.csv"
    with open(csv_file_sims, mode='w', newline='') as file_sims:
        df[['time', 'star1_x', 'star1_y', 'star1_z', 'star2_x', 'star2_y', 'star2_z']].to_csv(file_sims, index=False)


# Accuracy 0
def eb_acc_0(df:pd.DataFrame):
    pass

# Accuracy 1
def eb_acc_1(df:pd.DataFrame):
    for i in tqdm(range(len(df)), desc="Removing rows"):
        eclipse_score = df['eclipse_score'].iloc[i]
        primary_star = df['primary_star'].iloc[i]
        if eclipse_score > 0: # Score will be above 0 for partial, and 1 for full eclipsing
            if primary_star == 1:
                df['star2_x'], df['star2_y'], df['star2_z'] = None, None, None
            else:
                df['star1_x'], df['star1_y'], df['star1_z'] = None, None, None

# Accuracy 2
def eb_acc_2(df:pd.DataFrame):
    for i in tqdm(range(len(df)), desc="Removing rows"):
        eclipse_score = df['eclipse_score'].iloc[i]
        primary_star = df['primary_star'].iloc[i]
        if eclipse_score == 1: # Score will be one for full eclipsing
            if primary_star == 1:
                df['star2_x'], df['star2_y'], df['star2_z'] = None, None, None
            else:
                df['star1_x'], df['star1_y'], df['star1_z'] = None, None, None


def eb_setup(file_name:str, radius_star1:float, radius_star2:float, n = 4, add=False):
    """
    The set up is to calculate the position of evenly spaced boundary points along the circumference of the stars. It assumes main-sequence
    and spherical stars. It takes a circle in yz plane and rotate it to match the line of sight from origin. It is then projected onto a plane
    so that we can compare for boundary points

    Parameters:
    ----------
    file_name : str
        Name of the file containing the simulation data
    radius_star1 : float
        Radius of star 1 (main-sequence assumption)
    radius_star2 : float
        Radius of star 2 (main-sequence assumption)
    n : int 
        Number of boundary points on the circumference of the stars (input number as 2^n)
    add : bool
        Whether to add the boundary point positions to the sim file

    Returns:
    --------
    star1_boundary_points, star2_boundary_points : arrays of n arrays in each row
        Each row contains an array of [x, y, z] arrays of each boundary points 
        E.x if n = 4, for one row of star1_boundary_points, we have [[1,0,0],[0,1,0],[-1,0,0],[0,-1,0]]
    """

    df = pd.read_csv(f"scenarios/detailed_sims/{file_name}.csv")
    star1_r = radius_star1
    star2_r = radius_star2

    # Simulate as a circle with n points on the circumference, equally spaced out. Assumption that stars are perfectly spherical
    ang = 2*np.pi / n
    star1_boundary_points = []
    star2_boundary_points = []

    for i in tqdm(range(len(df)), desc="Setting up boundary points"):
        star1_pos = np.array([df['star1_x'].iloc[i], df['star1_y'].iloc[i], df['star1_z'].iloc[i]])
        n_pos = np.array([])

        # Calculate Rodrigues' rotation
        r = np.sqrt(star1_pos[0]**2 + star1_pos[1]**2 + star1_pos[2]**2)
        r_unit = star1_pos / r
        deg = np.arccos(r_unit[2]) # Angle between the star position and the x-axis of origin in xz plane
        R_x = rotate_about_axis([0,1,0], deg)
        
        phi = np.arccos(r_unit[1]) # Angle between the star position and the y-axis of origin in xy plane
        R_y = rotate_about_axis([0,0,1], phi)

        for j in range(n):
            theta = j * ang
            n_z = star1_pos[0] + star1_r * np.cos(theta)
            n_y = star1_pos[1] + star1_r * np.sin(theta)
            n_x = star1_pos[2]
            pos = np.array([n_x, n_y, n_z]) - star1_pos  # Subtract the star position to get the relative position
            pos = pos.reshape(3, 1)  # Reshape to a column vector
            pos = R_x @ pos  # Apply rotation matrix
            pos = R_y @ pos  # Apply rotation matrix
            pos = pos.flatten()  # Flatten back to a 1D array
            n_pos = np.append(n_pos, pos + star1_pos)  # Append the new position to the list
        
        star1_boundary_points.append(n_pos.reshape(n, 3))  # Reshape to a 2D array with n rows and 3 columns

        # Repeat for the second star
        star2_pos = np.array([df['star2_x'].iloc[i], df['star2_y'].iloc[i], df['star2_z'].iloc[i]])
        n_pos = np.array([])

        # Calculate Rodrigues' rotation
        r = np.sqrt(star2_pos[0]**2 + star2_pos[1]**2 + star2_pos[2]**2)
        r_unit = star2_pos / r
        deg = np.arccos(r_unit[2]) # Angle between the star position and the x-axis of origin in xz plane
        R_x = rotate_about_axis([0,1,0], deg)
        
        phi = np.arccos(r_unit[1]) # Angle between the star position and the y-axis of origin in xy plane
        R_y = rotate_about_axis([0,0,1], phi)

        for j in range(n):
            theta = j * ang
            n_z = star2_pos[0] + star2_r * np.cos(theta)
            n_y = star2_pos[1] + star2_r * np.sin(theta)
            n_x = star2_pos[2]
            pos = np.array([n_x, n_y, n_z]) - star2_pos  # Subtract the star position to get the relative position
            pos = pos.reshape(3, 1)  # Reshape to a column vector
            pos = R_x @ pos  # Apply rotation matrix
            pos = R_y @ pos  # Apply rotation matrix
            pos = pos.flatten()  # Flatten back to a 1D array
            n_pos = np.append(n_pos, pos + star2_pos)  # Append the new position to the list
        
        star2_boundary_points.append(n_pos.reshape(n, 3))  # Reshape to a 2D array with n rows and 3 columns

    if add:
        df['star1_boundary'] = list(star1_boundary_points)
        df['star2_boundary'] = list(star2_boundary_points)
        # Convert each (n,3) numpy array to a JSON string with commas
        df['star1_boundary'] = df['star1_boundary'].apply(lambda arr: json.dumps(arr.tolist()))
        df['star2_boundary'] = df['star2_boundary'].apply(lambda arr: json.dumps(arr.tolist()))

        # Now save to CSV
        df.to_csv(f"scenarios/detailed_sims/{file_name}.csv", index=False)
        return star1_boundary_points, star2_boundary_points
    else:
        return star1_boundary_points, star2_boundary_points


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
