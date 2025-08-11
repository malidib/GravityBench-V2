from tqdm import tqdm
import numpy as np

"""
This script is used to run the Phoebe model for a given set of parameters. The main purpose is to simulate a more realistic light curve
for a binary star system, as well as eclipsing binaries.
"""

import phoebe
from phoebe import u
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Update passbands to up-to-date accepted values
phoebe.update_all_passbands()

import math

def round_up(x, decimals=0):
    factor = 10 ** decimals
    return math.ceil(x * factor) / factor

class PhoebeModel:
    def __init__(self, file_name:str, multiprocessing=False, no_multiprocessing=0):
        self.file_name = file_name
        self.multiprocessing = multiprocessing
        self.no_multiprocessing = no_multiprocessing
        self.df = pd.read_csv(f"scenarios/detailed_sims/{file_name}.csv")
        self.df = self.df.copy(deep=True)
        df = self.df

        logger = phoebe.logger()

        if self.multiprocessing:
            print(f'Multiprocessing enabled, currently with {phoebe.multiprocessing_get_nprocs()} number of CPUs')
            phoebe.multiprocessing_on()
        
        if self.no_multiprocessing != 0:
            print(f'Multiprocessing enabled with {no_multiprocessing} CPUs enabled')
            phoebe.multiprocessing_set_nprocs(no_multiprocessing)

        Msun = 1.989e30
        Rsun = 6.957e8
        s_to_day = 1 / 86400 # Convert seconds to days

        df['time'] = df['time'] * s_to_day

        if os.path.exists(f'scenarios/phoebe_sims/{self.file_name}.json'):
            self.b = phoebe.Bundle.open(f'scenarios/phoebe_sims/{self.file_name}.json')
        else:
            # Start a new Phoebe bundle with the parameters from the Rebound file
            self.b = phoebe.Bundle()

            self.b.add_component(phoebe.component.star, component='primary')
            self.b.add_component(phoebe.component.star, component='secondary')

            self.b.add_orbit('binary')

            star1_m = (df['star1_mass'].iloc[0] / Msun) * u.solMass
            star2_m = (df['star2_mass'].iloc[0] / Msun) * u.solMass

            self.b.set_value(qualifier='mass', component='primary', context='component', value=star1_m)
            self.b.set_value(qualifier='mass', component='secondary', context='component', value=star2_m)

            #print(b['mass@primary@star@component'])

            star1_teff, star2_teff = self.teff(df)

            self.b.set_value(qualifier='teff', component='primary', context='component', value=star1_teff)
            self.b.set_value(qualifier='teff', component='secondary', context='component', value=star2_teff)
            
            self.b['incl@binary'] = df['inclination'].iloc[0]
            self.b['long_an@binary'] = df['longitude_of_ascending_node'].iloc[0]
            self.b['ecc@binary'] = df['eccentricity'].iloc[0]
            self.b['sma@binary'] = (df['semimajor_axis'].iloc[0] / Rsun) * u.solRad
            self.b['period@binary'] = (df['orbital_period'].iloc[0] * s_to_day) * u.d
            self.b['q@binary'] = star2_m / star1_m

            if star2_m > star1_m:
                # swap so the more massive one is primary
                self.b.set_hierarchy(phoebe.hierarchy.binaryorbit, self.b['binary'], self.b['secondary'], self.b['primary'])
            else:
                self.b.set_hierarchy(phoebe.hierarchy.binaryorbit, self.b['binary'], self.b['primary'], self.b['secondary'])

            # Set binary system distance (COM to origin)
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

            dis = np.sqrt(COMx**2 + COMy**2 + COMz**2)

            # Set distance to the system
            self.b.set_value('distance', dis * u.m)

    # Add light curve datasets
    def lc_compute(self):
        df = self.df

        # Find period
        period = df['orbital_period'].iloc[0] / 86400
        j = 0
        for i in df['time']:
            if i < period:
                j += 1
            else:
                break

        times = np.array(df['time'])[:j]
        print(period)
        self.b.add_dataset('lc', compute_times=times, dataset='lc01', overwrite=True)

        try:
            self.b.run_compute(irrad_method='none')
            print('run complete')
        except Exception as e:
            print(f"Error in chunk: {e}")
            self.b.run_failed_constraints()

        self.b.save(f"scenarios/phoebe_sims/{self.file_name}.json")

    def lc_graph(self):
        if 'lc01' not in self.b.datasets:
            self.lc_compute()

        afig, mplfig = self.b.plot(kind='lc', dataset='lc01', show=True)
        mplfig.savefig("scenarios/phoebe_output/lightcurve.png")


    def eb_compute(self):
        df = self.df

        # Find period
        period = df['orbital_period'].iloc[0] / 86400
        j = 0
        for i in df['time']:
            if i < period:
                j += 1
            else:
                break

        times = np.array(df['time'])[:j]

        # Run eclipse method
        phoebe.devel_on()

        self.b.add_dataset('mesh', compute_times=times, columns=['visibilities'], dataset='mesh01', overwrite=True)

        try:
            self.b.run_compute(irrad_method='none',
                        eclipse_method='visible_partial',
                        mesh_method='wd')
            print('Run complete')

            # Get twig names for visibility params
            print(self.b.filter(qualifier='visibilities',
                            component='primary',
                            dataset='mesh01',
                            model='latest',
                            context='model'))

        except Exception as e:
            print(f"Error in chunk: {e}")
            self.b.run_failed_constraints()

        self.b.save(f"scenarios/phoebe_sims/{self.file_name}.json")

    def eb_setup(self):
        df = self.df

        if 'mesh01' not in self.b.datasets:
            self.eb_compute()

        # Find period
        period = df['orbital_period'].iloc[0] / 86400
        j = 0
        for i in df['time']:
            if i < period:
                j += 1
            else:
                break

        times = np.array(df['time'])[:j]

        # Check values, 0 not visible, 0.5 partially visible, 1 visible
        star1_vis = self.b.get_value().mean() ## Find out how to call from datasets
        star2_vis = self.b.get_value().mean()

        star1_vis_col = np.array([])
        if star1_vis > 0:
            star1_vis_col = np.append(star1_vis_col, 1)
        else:
            star1_vis_col.np.append(star2_vis_col, 0)
        
        star2_vis_col = np.array([])
        if star2_vis > 0:
            star2_vis_col = np.append(star2_vis_col, 1)
        else:
            star2_vis_col = np.append(star2_vis_col, 0)
        
        df['star1_vis'] = star1_vis_col
        df['star2_vis'] = star2_vis_col

        for i in range(len(times)):
            if df['star1_vis'].iloc[i] == 0:
                df['star1_x'].iloc[i], df['star1_y'].iloc[i], df['star1_z'].iloc[i] = None, None, None
            if df['star2_vis'].iloc[i] == 0:
                df['star2_x'].iloc[i], df['star2_y'].iloc[i], df['star2_z'].iloc[i] = None, None, None


    def teff(self, df):
        # Calculate effective temperature based on the mass
        Msun = 1.989e30
        sb_const = 5.67e-8
        Lsun = 3.846e26
        Rsun = 6.957e8

        star1_m = df['star1_mass'].iloc[0]
        star1_teff = (((star1_m/Msun)**1.6)*Lsun/(4*np.pi*sb_const*(Rsun**2)))**(1/4)

        star2_m = df['star2_mass'].iloc[0]
        star2_teff = (((star2_m/Msun)**3.5)*Lsun/(4*np.pi*sb_const*(Rsun**2)))**(1/4)

        return star1_teff, star2_teff

    

# = PhoebeModel(file_name='3.2 M, 2.1 M, PRO', multiprocessing=True)
#m.lc_graph()
