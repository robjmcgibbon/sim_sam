import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

helpers_path = os.path.abspath(sys.path[0]+'/..')
sys.path.append(helpers_path)
import helpers
from helpers import log

snaps = [33, 50, 99]  # Don't use snaps lower than the cut from extract_trees
use_trees = True

config = helpers.Config()
h = config.hubble_param
omega_m = config.omega_m
omega_b = config.omega_b
colors = config.get_colors()
names = config.get_names()

log(f'Loading data')
data = {snap: {} for snap in snaps}
if use_trees:
    data_dir = config.get_generated_data_dir() + 'trees/'
    trees = {filename.replace('.npy', ''): np.load(data_dir+filename) for filename in os.listdir(data_dir)}
    for snap in snaps:
        for efficiency in ['f_a', 'f_a_id', 'f_c', 'f_s', 'f_d', 'f_m']:
            data[snap][efficiency] = trees[efficiency][:, snap]
        data[snap]['dm_mass'] = trees['dm_mass'][:, snap]
else:
    data_dir = config.get_generated_data_dir() + 'efficiencies/'
    for snap in snaps:
        d = pd.read_parquet(f'{data_dir}snap_{snap}.parquet')
        data[snap] = {k: np.array(d[k]) for k in d.keys()}
        data[snap]['dm_mass'] = data[snap]['desc_dm_mass']

bins = np.linspace(11, 14, 10)
mids = (bins[1:] + bins[:-1]) / 2
for efficiency in [
        'f_a',
        'f_c',
        'f_s',
        'f_d',
        'f_m',
        ]:
    fig, ax = plt.subplots(1)
    for snap in snaps:
        arr_dm_mass = np.log10(data[snap]['dm_mass']) + 10  # Fixing units
        vals = np.zeros(mids.shape[0])
        for i_bin in range(mids.shape[0]):
            low_lim, upp_lim = bins[i_bin], bins[i_bin+1]
            mask = arr_dm_mass > low_lim
            mask &= arr_dm_mass < upp_lim
            mask &= data[snap][efficiency] != -1
            vals[i_bin] = np.mean(data[snap][efficiency][mask])
        mask = vals != 0
        z = config.get_redshifts()[snap]
        ax.plot(mids[mask], vals[mask], color=colors[z], label=f'z={z}')

    ax.set_title(efficiency)
    ax.set_xlabel('Halo mass [$M_\odot$ / h]')
    if 'f_a' in efficiency:
        universal_baryon_fraction = omega_b / (omega_m - omega_b)
        ax.axhline(universal_baryon_fraction, ls='--', color='gray', label=r'$\frac{\Omega_b}{\Omega_m-\Omega_b}$')
    ax.legend()
    plt.savefig(f'/home/rmcg/ss_{efficiency}_vs_mass.png', dpi=200)
    plt.close()
