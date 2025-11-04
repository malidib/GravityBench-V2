import os, glob, sys
from tqdm import tqdm
import pandas as pd
import sys, os
V2_root = "/Users/jaredlim/Desktop/GravityBenchV2"
sys.path.append(V2_root)
import scripts.geometry_config as geometry_config

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
        geometry_config.projection(pd.read_csv(f"{d}"), stem, save=True) # Create the projection dataframe
        proj_file = os.path.join(proj_dir, f"{stem}.csv")
        stem_to_proj[stem] = proj_file  # Update dict
    file_names.append((d, stem_to_proj[stem]))

max_list = []
for i in tqdm(projected):
    new_df = pd.read_csv(i)
    old_df = pd.read_csv(f"/Users/jaredlim/Desktop/projected_sims_copy/{i.split('/')[-1]}")

    max_diff = 0
    threshold_value = 1000
    for i in range(len(new_df)):
        df_row = old_df.iloc[i]
        test_row = new_df.iloc[i]
        current_max_diff = max(abs(df_row['star1_x'] - test_row['star1_x']), 
                                abs(df_row['star1_y'] - test_row['star1_y']),
                                abs(df_row['star1_z'] - test_row['star1_z']), 
                                abs(df_row['star2_x'] - test_row['star2_x']),
                                abs(df_row['star2_y'] - test_row['star2_y']), 
                                abs(df_row['star2_z'] - test_row['star2_z']),
                                )
        if current_max_diff > max_diff:
            max_diff = current_max_diff
    
    max_list.append(max_diff)

print(max_list)
