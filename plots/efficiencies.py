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
colors = config.get_colors()

log(f'Loading data')
data_dir = config.get_generated_data_dir() + 'efficiencies/'
data = {}
for snap in snaps:
    d = pd.read_parquet(f'{data_dir}snap_{snap}.parquet')
    data[snap] = {k: np.array(d[k]) for k in d.keys()}

for efficiency, bins in [
        ('f_a', np.linspace(0, 1, 51)),
        ('f_c', np.linspace(0, 0.05, 51)),
        ('f_s', np.linspace(0, 0.5, 51)),
        ('f_d', np.linspace(0, 10, 51))
        ]:
    log(f'Efficiency: {efficiency}')
    fig, ax = plt.subplots(1, dpi=150)
    for snap in snaps:
        z = round(config.get_redshifts()[snap], 1)

        mask = np.array(data[snap][efficiency]) != -1
        frac_valid = np.sum(mask) / mask.shape[0]
        frac_zero = np.sum(data[snap][efficiency][mask] == 0) / mask.shape[0]
        log(f'z={z}: n_valid={np.sum(mask)}, frac_valid={frac_valid:.3g}, frac_zero={frac_zero:.3g}')

        mean = np.mean(data[snap][efficiency][mask])
        ax.axvline(mean, linestyle='dashed', color=colors[z])

        ax.hist(data[snap][efficiency][mask], label=f'z={z}',
                histtype='step', color=colors[z],
                density=True, bins=bins)

    ax.set_xlabel(f'${efficiency}$', fontsize=14)
    ax.legend()
    
    # plt.savefig(f'/home/rmcg/ss_hist_{efficiency}.png')
    plt.show()
    plt.close()
