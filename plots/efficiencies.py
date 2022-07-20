import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

helpers_path = os.path.abspath(sys.path[0]+'/..')
sys.path.append(helpers_path)
import helpers
from helpers import log

snaps = [25, 33, 50, 99]
config = helpers.Config()
h = config.hubble_param
omega_m = config.omega_m
omega_b = config.omega_b

log(f'Loading data')
data_dir = config.get_generated_data_dir() + 'efficiencies/'
data = {}
for snap in snaps:
    data[snap] = pd.read_parquet(f'{data_dir}snap_{config.snap}.parquet')

print(data)
exit()
if not os.path.exists(save_data_dir):
    os.makedirs(save_data_dir)
pd.DataFrame(data).to_parquet(f'{save_data_dir}snap_{config.snap}.parquet', index=False)

z0_data = pd.read_csv('snap_98.csv', index_col=0)
z1_data = pd.read_csv('snap_49.csv', index_col=0)
z2_data = pd.read_csv('snap_32.csv', index_col=0)
colors = ['tab:blue', 'tab:orange', 'tab:green']

# TODO: Fraction outside of bins
# TODO: Print fraction invalid
for efficiency, bins in [
        ('f_a', np.linspace(0, 1, 51)),
        ('f_c', np.linspace(0, 0.05, 51)),
        ('f_s', np.linspace(0, 0.5, 51)),
        ('f_d', np.linspace(0, 10, 51))
        ]:
    print(f'Efficiency: {efficiency}')
    fig, ax = plt.subplots(1, dpi=150)
    for z, data in enumerate([z0_data, z1_data, z2_data]):
        # TODO: What mask to use?
        # mask = data['sub_stellar_mass'] > 0.1
        # data = data[mask]

        mask = np.array(data[efficiency]) != -1
        frac_valid = np.sum(mask) / mask.shape[0]
        mean = np.mean(data[efficiency][mask])
        ax.axvline(mean, linestyle='dashed', color=colors[z])
        ax.hist(data[efficiency][mask], label=f'z={z}',
                histtype='step', color=colors[z],
                density=True, bins=bins)
        frac_zero = np.sum(data[efficiency][mask] == 0) / mask.shape[0]
        print(f'z={z}: n_valid={np.sum(mask)}, frac_valid={frac_valid:.3g}, frac_zero={frac_zero:.3g}')


    # ax.text(0.5, 0.8, f'Fraction valid: {np.sum(mask)/mask.shape[0]:.2g}',
            # transform=ax.transAxes, fontsize=14)
    # ax.text(0.5, 0.7, f'Mean: {np.mean(data[efficiency][mask]):.2g}',
            # transform=ax.transAxes, fontsize=14)
    # ax.set_title(f'{sim.upper()}, z={round(redshifts[snap])}', fontsize=18)
    ax.set_xlabel(f'${efficiency}$', fontsize=14)
    ax.legend()
    
    # plt.savefig(f'/home/rmcg/{sim}_{efficiency}_hist_snap{snap}.png')
    plt.show()
    plt.close()
