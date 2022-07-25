import os

import matplotlib.pyplot as plt
import numpy as np

import helpers
from helpers import log

config = helpers.Config()
ages = config.get_ages()
colors = config.get_colors()

log(f'Loading data')
tree_data_dir = config.get_generated_data_dir() + 'trees/'

trees = {filename.replace('.npy', ''): np.load(tree_data_dir+filename) for filename in os.listdir(tree_data_dir)}
n_tree = trees['subhalo_id'].shape[0]


# TODO: Use both efficiences from actual galaxy, and use mean efficiencies
def plot_galaxy(i_tree):
    dm_mass = trees['dm_mass'][i_tree, :]
    hot_gas_mass = trees['hot_gas_mass'][i_tree, :]
    cold_gas_mass = trees['cold_gas_mass'][i_tree, :]
    stellar_mass = trees['stellar_mass'][i_tree, :]

    fig, ax = plt.subplots(1)
    ax.plot(ages[dm_mass != 0], dm_mass[dm_mass != 0], label='DM mass', color=colors['dm_mass'])
    ax.plot(ages[hot_gas_mass != 0], hot_gas_mass[hot_gas_mass != 0],
            label='Hot gas mass', color=colors['hot_gas_mass'])
    ax.plot(ages[cold_gas_mass != 0], cold_gas_mass[cold_gas_mass != 0],
            label='Cold gas mass', color=colors['cold_gas_mass'])
    ax.plot(ages[stellar_mass != 0], stellar_mass[stellar_mass != 0],
            label='Stellar mass', color=colors['stellar_mass'])

    sam_dm_mass = np.zeros(100, dtype='float64')
    sam_hot_gas_mass = np.zeros(100, dtype='float64')
    sam_cold_gas_mass = np.zeros(100, dtype='float64')
    sam_stellar_mass = np.zeros(100, dtype='float64')

    i = np.where(dm_mass != 0)[0][0]
    sam_dm_mass[i] = dm_mass[i]
    sam_hot_gas_mass[i] = hot_gas_mass[i]
    sam_cold_gas_mass[i] = cold_gas_mass[i]
    sam_stellar_mass[i] = stellar_mass[i]

    diff_dm_mass = trees['diff_dm_mass'][i_tree, :]
    f_a = np.minimum(trees['f_a'][i_tree, :], 1)
    f_c = np.minimum(trees['f_c'][i_tree, :], 1)
    f_s = np.minimum(trees['f_s'][i_tree, :], 1)

    # TODO: Interpolate invalid rates
    i += 1
    while i < 100:
        sam_dm_mass[i] = sam_dm_mass[i-1]
        sam_hot_gas_mass[i] = sam_hot_gas_mass[i-1]
        sam_cold_gas_mass[i] = sam_cold_gas_mass[i-1]
        sam_stellar_mass[i] = sam_stellar_mass[i-1]

        sam_dm_mass[i] += diff_dm_mass[i]
        sam_hot_gas_mass[i] += f_a[i] * diff_dm_mass[i]

        if f_c[i] != -1:
            sam_hot_gas_mass[i] -= f_c[i] * sam_hot_gas_mass[i-1]
            sam_cold_gas_mass[i] += f_c[i] * sam_hot_gas_mass[i-1]

        if f_s[i] != -1:
            sam_cold_gas_mass[i] -= f_s[i] * sam_cold_gas_mass[i-1]
            sam_stellar_mass[i] += f_s[i] * sam_cold_gas_mass[i-1]

        # TODO: Set minimum values for properties
        sam_cold_gas_mass[i] = np.maximum(sam_cold_gas_mass[i], 0)

        i += 1

    ax.plot(ages[sam_dm_mass != 0], sam_dm_mass[sam_dm_mass != 0], '--', color=colors['dm_mass'])
    ax.plot(ages[sam_hot_gas_mass != 0], sam_hot_gas_mass[sam_hot_gas_mass != 0], '--', color=colors['hot_gas_mass'])
    ax.plot(ages[sam_cold_gas_mass != 0], sam_cold_gas_mass[sam_cold_gas_mass != 0], '--', color=colors['cold_gas_mass'])
    ax.plot(ages[sam_stellar_mass != 0], sam_stellar_mass[sam_stellar_mass != 0], '--', color=colors['stellar_mass'])

    ax.plot([0], [0], 'k-', label='Simulation')
    ax.plot([0], [0], 'k--', label='Model')

    ax.set_xlabel('Universe age')
    ax.set_ylabel('Mass [$10^{10}M_\odot$]')
    ax.set_xlim(0.3, 14)
    ax.set_yscale('log')

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(ax.get_xticks())
    ax2_xlabels = []
    for age in ax.get_xticks():
        closest_snap = config.get_closest_snapshot_for_age(age)
        closest_redshift = config.get_redshifts()[closest_snap]
        closest_redshift = round(closest_redshift, 1)
        ax2_xlabels.append(closest_redshift)
    ax2.set_xticklabels(ax2_xlabels)
    ax2.set_xlabel('z')

    ax.legend()
    plt.savefig(f'/home/rmcg/galaxy_{i_tree}.png', dpi=200)
    plt.close()


while j < 1:
    if trees['dm_mass'][i, 50] != 0:
        plot_galaxy(i)
        j += 1
    i += 1



# efficiencies = ['f_a', 'f_c', 'f_s', 'f_d', 'f_m']
# data = {e: {} for e in efficiencies}
# for snap in range(100):
#     valid = trees[:, 3, snap] != 0   # Check for nonzero dm_mass
#     data['f_a'][snap] = trees[:, 14, snap][valid]
#     data['f_c'][snap] = trees[:, 15, snap][valid]
#     data['f_s'][snap] = trees[:, 16, snap][valid]
#     data['f_d'][snap] = trees[:, 17, snap][valid]
#     data['f_m'][snap] = trees[:, 18, snap][valid]
#
# median = {e: np.zeros(100) for e in efficiencies}
# lower = {e: np.zeros(100) for e in efficiencies}
# upper = {e: np.zeros(100) for e in efficiencies}
# for e in efficiencies:
#     for snap in range(100):
#         valid = data[e][snap] != -1
#         print(f'Efficiency: {e}, Snap: {snap}, n_valid: {np.sum(valid)}')
#         if np.sum(valid) != 0:
#             median[e][snap] = np.mean(data[e][snap][valid])
#             # median[e][snap] = np.percentile(data[e][snap][valid], 50)
#             lower[e][snap] = np.percentile(data[e][snap][valid], 25)
#             upper[e][snap] = np.percentile(data[e][snap][valid], 75)
#
# ages = config.get_ages()
# fig, ax = plt.subplots(1)
# for e in efficiencies:
#     p = ax.plot(ages, median[e], '-', label=e)
#     # ax.plot(ages, lower[e], '--', color=p[0].get_color())
#     # ax.plot(ages, upper[e], '--', color=p[0].get_color())
# ax.legend()
# ax.set_yscale('log')
# ax.set_xlabel('Universe age')
# ax.set_xlim(0.3, 14)
#
# ax2 = ax.twiny()
# ax2.set_xlim(ax.get_xlim())
# ax2.set_xticks(ax.get_xticks())
# ax2_xlabels = []
# for age in ax.get_xticks():
#     closest_snap = config.get_closest_snapshot_for_age(age)
#     closest_redshift = config.get_redshifts()[closest_snap]
#     closest_redshift = round(closest_redshift, 1)
#     ax2_xlabels.append(closest_redshift)
# ax2.set_xticklabels(ax2_xlabels)
# ax2.set_xlabel('z')
# plt.savefig(f'/home/rmcg/ss_efficiences_vs_time.png', dpi=200)
# plt.close()

log(f'Script finished')

