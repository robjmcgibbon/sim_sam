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

for efficiency, bins, cut in [
        ('f_a', np.linspace(0, 1, 51), 1),
        ('f_a_id', np.linspace(0, 1, 51), 1),
        ('f_c', np.linspace(0, 0.5, 51), 1),
        ('f_s', np.linspace(0, 1, 51), 1),
        ('f_d', np.linspace(0, 10, 51), float('inf')),
        ('f_m', np.linspace(0, 10, 51), float('inf'))
        ]:
    log(f'Efficiency: {efficiency}')
    fig, ax = plt.subplots(1, dpi=150)
    for snap in snaps:
        z = round(config.get_redshifts()[snap], 1)

        mask = data[snap][efficiency] != -1
        frac_valid = np.sum(mask) / mask.shape[0]

        cut_mask = data[snap][efficiency] > cut
        frac_valid_above_cut = np.sum(cut_mask) / np.sum(mask)
        max_value = np.max(data[snap][efficiency])
        data[snap][efficiency][cut_mask] = cut

        frac_valid_zero = np.sum(data[snap][efficiency] == 0) / np.sum(mask)

        log(f'z={z}: n_valid={np.sum(mask)}, frac_valid={frac_valid:.3g}, frac_valid_zero={frac_valid_zero:.3}',
            f'cut={cut}, frac_valid_above_cut={frac_valid_above_cut:.3g}, max={max_value}')

        mean = np.mean(data[snap][efficiency][mask])
        ax.axvline(mean, linestyle='dashed', color=colors[z])

        ax.hist(data[snap][efficiency][mask], label=f'z={z}',
                histtype='step', color=colors[z],
                density=True, bins=bins)

    ax.set_xlabel(f'${efficiency}$', fontsize=14)
    ax.legend()
    if 'f_a' in efficiency:
        universal_baryon_fraction = omega_b / (omega_m - omega_b)
        ax.axvline(universal_baryon_fraction, ls='--', color='gray', label=r'$\frac{\Omega_b}{\Omega_m-\Omega_b}$')

    # plt.savefig(f'/home/rmcg/ss_hist_{efficiency}.png')
    plt.show()
    plt.close()

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
    plt.savefig(f'/home/rmcg/ss_{efficiency}_vs_dm_mass.png', dpi=200)
    plt.close()
