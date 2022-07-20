import os
import multiprocessing

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import helpers
from helpers import log

config = helpers.Config()

log(f'Loading data')
tree_data_dir = config.get_generated_data_dir()
trees = np.load(f'{tree_data_dir}trees.npy')
n_tree = trees.shape[0]

# TODO: Don't have to copy this across from extract_trees.py
input_properties = ['bh_mass', 'bh_dot', 'cold_gas_mass', 'dm_mass', 'hot_gas_mass',
                    'gas_metallicity', 'sfr', 'stellar_mass', 'stellar_metallicity',
                    'rate_accrete_hot', 'rate_hot_cold', 'rate_cold_stars',
                    'rate_cold_hot', 'rate_accrete_stars',
                    'f_a', 'f_c', 'f_s', 'f_d', 'f_m']
assert len(input_properties) == trees.shape[1]

efficiencies = ['f_a', 'f_c', 'f_s', 'f_d', 'f_m']
data = {e: {} for e in efficiencies}
for snap in range(100):
    valid = trees[:, 3, snap] != 0   # Check for nonzero dm_mass
    data['f_a'][snap] = trees[:, 14, snap][valid]
    data['f_c'][snap] = trees[:, 15, snap][valid]
    data['f_s'][snap] = trees[:, 16, snap][valid]
    data['f_d'][snap] = trees[:, 17, snap][valid]
    data['f_m'][snap] = trees[:, 18, snap][valid]

median = {e: np.zeros(100) for e in efficiencies}
lower = {e: np.zeros(100) for e in efficiencies}
upper = {e: np.zeros(100) for e in efficiencies}
for e in efficiencies:
    for snap in range(100):
        valid = data[e][snap] != -1
        print(f'Efficiency: {e}, Snap: {snap}, n_valid: {np.sum(valid)}')
        if np.sum(valid) != 0:
            median[e][snap] = np.mean(data[e][snap][valid])
            # median[e][snap] = np.percentile(data[e][snap][valid], 50)
            lower[e][snap] = np.percentile(data[e][snap][valid], 25)
            upper[e][snap] = np.percentile(data[e][snap][valid], 75)

ages = config.get_ages()
fig, ax = plt.subplots(1)
for e in efficiencies:
    p = ax.plot(ages, median[e], '-', label=e)
    # ax.plot(ages, lower[e], '--', color=p[0].get_color())
    # ax.plot(ages, upper[e], '--', color=p[0].get_color())
ax.legend()
ax.set_yscale('log')
ax.set_xlabel('Universe age')
ax.set_xlim(0.3, 14)

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
plt.savefig(f'/home/rmcg/ss_efficiences_vs_time.png', dpi=200)
plt.close()

bins = np.linspace(11, 14, 10)
mids = (bins[1:] + bins[:-1]) / 2
for e, i_e in [('f_a', 14), ('f_c', 15), ('f_s', 16), ('f_d', 17), ('f_m', 18)]:
    fig, ax = plt.subplots(1)
    for snap in [33, 50, 99]:
        arr_dm_mass = np.log10(trees[:, 3, snap] + 0.000001) + 10  # Very hacky
        arr_e = trees[:, i_e, snap]
        vals = np.zeros(mids.shape[0])
        for i_bin in range(mids.shape[0]):
            low_lim, upp_lim = bins[i_bin], bins[i_bin+1]
            mask = arr_dm_mass > low_lim
            mask &= arr_dm_mass < upp_lim
            mask &= arr_e != -1
            vals[i_bin] = np.mean(arr_e[mask])
        mask = vals != 0
        ax.plot(mids[mask], vals[mask], label=f'z={config.get_redshifts()[snap]}')
    ax.set_title(e)
    ax.set_xlabel('DM mass')
    ax.legend()
    plt.savefig(f'/home/rmcg/ss_{e}_vs_dm_mass.png', dpi=200)
    plt.close()


def plot_galaxy(i_tree):
    fig, ax = plt.subplots(1)
    dm_mass = trees[i_tree, 3, :]
    cold_gas_mass = trees[i_tree, 2, :]
    hot_gas_mass = trees[i_tree, 4, :]
    stellar_mass = trees[i_tree, 7, :]
    arr_f_a = trees[i_tree, 14, :]
    arr_f_c = trees[i_tree, 15, :]
    arr_f_s = trees[i_tree, 16, :]
    arr_f_d = trees[i_tree, 17, :]
    # TODO: Include f_m

    ax.plot(ages[dm_mass!=0], dm_mass[dm_mass!=0], label='DM mass', color='tab:blue')
    ax.plot(ages[cold_gas_mass!=0], cold_gas_mass[cold_gas_mass!=0], label='Cold gas mass', color='tab:green')
    ax.plot(ages[hot_gas_mass!=0], hot_gas_mass[hot_gas_mass!=0], label='Hot gas mass', color='tab:red')
    ax.plot(ages[stellar_mass!=0], stellar_mass[stellar_mass!=0], label='Stellar mass', color='tab:orange')

    start = np.where(dm_mass!=0)[0][0]
    sam_cold_gas_mass = [0] * start + [cold_gas_mass[start]]
    sam_hot_gas_mass = [0] * start + [hot_gas_mass[start]]
    sam_stellar_mass = [0] * start + [stellar_mass[start]]
    print(sam_cold_gas_mass)
    print(sam_hot_gas_mass)
    print(sam_stellar_mass)
    start += 1
    while start < 100:
        diff_dm_mass = dm_mass[start] - dm_mass[start - 1]
        cold = sam_cold_gas_mass[-1]
        hot = sam_hot_gas_mass[-1]
        star = sam_stellar_mass[-1]
        f_a = arr_f_a[start]
        f_c = arr_f_c[start]
        f_s = arr_f_s[start]
        f_d = arr_f_d[start]

        if f_s != -1:
            star += f_s * sam_cold_gas_mass[-1]

        if f_a != -1:
            # TODO: Major hack
            hot += f_a * diff_dm_mass * 0.1
        if f_c != -1:
            hot -= f_c * sam_hot_gas_mass[-1]
        # if (f_d != -1) and (f_s != -1):
        #     hot += f_d * f_s * sam_cold_gas_mass[-1]

        if f_c != -1:
            cold += f_c * sam_hot_gas_mass[-1]
        if f_s != -1:
            cold -= f_s * sam_cold_gas_mass[-1]
        # if (f_d != -1) and (f_s != -1):
        #     cold -= f_d * f_s * sam_cold_gas_mass[-1]

        sam_cold_gas_mass.append(cold)
        sam_hot_gas_mass.append(hot)
        sam_stellar_mass.append(star)

        start += 1

    sam_cold_gas_mass = np.array(sam_cold_gas_mass)
    sam_hot_gas_mass = np.array(sam_hot_gas_mass)
    sam_stellar_mass = np.array(sam_stellar_mass)
    print(sam_cold_gas_mass)
    print(sam_hot_gas_mass)
    print(sam_stellar_mass)
    print()
    ax.plot(ages[sam_cold_gas_mass!=0], sam_cold_gas_mass[sam_cold_gas_mass!=0], '--', color='tab:green')
    ax.plot(ages[sam_hot_gas_mass!=0], sam_hot_gas_mass[sam_hot_gas_mass!=0], '--', color='tab:red')
    ax.plot(ages[sam_stellar_mass!=0], sam_stellar_mass[sam_stellar_mass!=0], '--', color='tab:orange')

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

for i in range(5):
    plot_galaxy(i)


# TODO: Plot value for single galaxy over time

# TODO: Plot evolution of gas, stellar over time
# TODO: Use both efficiences from actual galaxy, and use mean efficiencies



exit()

n_process = 5

config = helpers.Config()
log(f'Extracting tree data for {config.name} for snapshot {config.snap}')
lhalotree_dir = config.get_lhalotree_dir()


def extract_trees(filepath):
    log(f'Starting processing file: {filepath}')
    with h5py.File(filepath, 'r') as file:
        n_halos_in_tree = np.array(file['/Header/TreeNHalos'])

        trees = []
        for i_tree, n_halo in enumerate(n_halos_in_tree):
            arr = {}
            tree = file[f'Tree{i_tree}']

            # Convert mass to solar units
            arr_mass_type = np.array(tree['SubhaloMassType'])
            arr['bh_mass'] = np.array(tree['SubhaloBHMass'])
            arr['bh_dot'] = np.array(tree['SubhaloBHMdot'])
            arr['dm_mass'] = arr_mass_type[:, 1]
            arr['gas_mass'] = arr_mass_type[:, 0]
            arr['gas_metallicity'] = np.array(tree['SubhaloGasMetallicity'])
            arr['sfr'] = np.array(tree['SubhaloSFR'])
            arr['stellar_mass'] = arr_mass_type[:, 4]
            arr['stellar_metallicity'] = np.array(tree['SubhaloStarMetallicity'])

            arr['main_prog_index'] = np.array(tree['FirstProgenitor'])
            arr['snap_num'] = np.array(tree['SnapNum'])
            arr['subhalo_id'] = np.array(tree['SubhaloNumber'])

            arr_central_index = np.array(tree['FirstHaloInFOFGroup'])
            arr['is_central'] = np.zeros(n_halo, dtype=bool)
            for i_halo, i_central in enumerate(arr_central_index):
                arr['is_central'][i_halo] = (i_halo == i_central)

            min_snap = 2
            max_snap = 99
            input_properties = ['bh_mass', 'bh_dot', 'dm_mass', 'gas_mass', 'gas_metallicity',
                                'sfr', 'stellar_mass', 'stellar_metallicity']

            snapshots = list(range(max_snap, min_snap-1, -1))
            n_input, n_snap = len(input_properties), len(snapshots)
            input_features = [str(snap)+prop for snap in snapshots for prop in input_properties]

            # Applying same criteria as in extract_pairs
            valid = (arr['snap_num'] == max_snap)
            valid &= (arr['dm_mass'] > config.dm_mass_cut)
            valid &= arr['is_central']
            valid &= (arr['gas_mass'] != 0)
            valid &= (arr['stellar_mass'] != 0)
            n_valid_sub_this_file = np.sum(valid)
            if n_valid_sub_this_file == 0:
                continue

            i_sub = 0
            histories = np.zeros((n_valid_sub_this_file, n_input*n_snap), dtype='float64')
            for i_halo in np.arange(n_halo)[valid]:
                i_prog = i_halo
                while i_prog != -1:
                    snap_num = arr['snap_num'][i_prog]
                    if snap_num < min_snap:
                        break

                    bh_mass = arr['bh_mass'][i_prog]
                    bh_dot = arr['bh_dot'][i_prog]
                    dm_mass = arr['dm_mass'][i_prog]
                    gas_mass = arr['gas_mass'][i_prog]
                    gas_metallicity = arr['gas_metallicity'][i_prog]
                    sfr = arr['sfr'][i_prog]
                    stellar_mass = arr['stellar_mass'][i_prog]
                    stellar_metallicity = arr['stellar_metallicity'][i_prog]

                    i_start = (max_snap - snap_num) * n_input
                    # This has to line up with where input columns are defined
                    data = [bh_mass, bh_dot, dm_mass, gas_mass, gas_metallicity,
                            sfr, stellar_mass, stellar_metallicity]
                    histories[i_sub, i_start:i_start+n_input] = data

                    i_prog = arr['main_prog_index'][i_prog]

                i_sub += 1

            trees.append(pd.DataFrame(histories, columns=input_features))
    return pd.concat(trees, ignore_index=True)


all_histories = []
pool = multiprocessing.Pool(n_process)
filenames = [lhalotree_dir+name for name in os.listdir(lhalotree_dir)]
while filenames:
    files_to_process, filenames = filenames[:n_process], filenames[n_process:]
    pool_result = pool.map(extract_trees, files_to_process)

    log('Concatenating dataframes')
    if type(all_histories) == list:
        all_histories = pool_result.pop(0)
    while pool_result:
        all_histories = pd.concat([all_histories, pool_result.pop(0)], ignore_index=True)

log(f'Saving data')
save_data_dir = config.get_generated_data_dir()
if not os.path.exists(save_data_dir):
    os.makedirs(save_data_dir)
all_histories.to_parquet(f'{save_data_dir}trees.parquet', index=False)

log(f'Script finished')
