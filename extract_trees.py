import os
import multiprocessing

import h5py
import numpy as np
import pandas as pd

import helpers
from helpers import log

n_process = 5

config = helpers.Config()
log(f'Extracting tree data for {config.name} for snapshot {config.snap}')
lhalotree_dir = config.get_lhalotree_dir()


def extract_trees(filepath):
    log(f'Starting processing file: {filepath}')
    with h5py.File(filepath, 'r') as file:
        n_halos_in_tree = np.array(file['/Header/TreeNHalos'])

        # TODO: Hot and cold gas metallicity?
        input_properties = ['bh_mass', 'bh_dot', 'cold_gas_mass', 'dm_mass', 'hot_gas_mass',
                            'gas_metallicity', 'sfr', 'stellar_mass', 'stellar_metallicity',
                            'rate_accrete_hot', 'rate_hot_cold', 'rate_cold_stars',
                            'rate_cold_hot', 'rate_accrete_stars',
                            'f_a', 'f_c', 'f_s', 'f_d', 'f_m']

        min_snap = 2
        max_snap = 99
        n_input, n_snap = len(input_properties), max_snap+1

        # TODO: Save as dict of numpy arrays rather than full array
        trees = np.zeros((0, n_input, n_snap), dtype='float64')
        subhalo_ids = np.zeros((0, n_snap), dtype='int64')
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
            histories = np.zeros((n_valid_sub_this_file, n_input, n_snap), dtype='float64')
            tree_subhalo_ids = -1 * np.ones((n_valid_sub_this_file, n_snap), dtype='int64')
            for i_halo in np.arange(n_halo)[valid]:
                i_prog = i_halo
                while i_prog != -1:
                    snap_num = arr['snap_num'][i_prog]
                    if snap_num < min_snap:
                        break

                    histories[i_sub, 0, snap_num] = arr['bh_mass'][i_prog]
                    histories[i_sub, 1, snap_num] = arr['bh_dot'][i_prog]
                    histories[i_sub, 3, snap_num] = arr['dm_mass'][i_prog]
                    # TODO: Remove?
                    histories[i_sub, 4, snap_num] = arr['gas_mass'][i_prog]
                    histories[i_sub, 5, snap_num] = arr['gas_metallicity'][i_prog]
                    histories[i_sub, 6, snap_num] = arr['sfr'][i_prog]
                    histories[i_sub, 7, snap_num] = arr['stellar_mass'][i_prog]
                    histories[i_sub, 8, snap_num] = arr['stellar_metallicity'][i_prog]

                    tree_subhalo_ids[i_sub, snap_num] = arr['subhalo_id'][i_prog]

                    i_prog = arr['main_prog_index'][i_prog]

                i_sub += 1

            trees = np.concatenate([trees, histories], axis=0)
            subhalo_ids = np.concatenate([subhalo_ids, tree_subhalo_ids], axis=0)
    return trees, subhalo_ids


pool_result = []
pool = multiprocessing.Pool(n_process)
filenames = [lhalotree_dir+name for name in os.listdir(lhalotree_dir)]
while filenames:
    files_to_process, filenames = filenames[:n_process], filenames[n_process:]
    pool_result += pool.map(extract_trees, files_to_process)

all_trees, all_subhalo_ids = pool_result.pop(0)
for pool_trees, pool_subhalos_ids in pool_result:
    all_trees = np.concatenate([all_trees, pool_trees], axis=0)
    all_subhalo_ids = np.concatenate([all_subhalo_ids, pool_subhalos_ids], axis=0)

efficiencies_data_dir = config.get_generated_data_dir() + 'efficiencies/'
for snap in range(1, 100):
    log(snap)
    efficiencies = pd.read_parquet(f'{efficiencies_data_dir}snap_{snap}.parquet')

    dict_gas_mass = {}
    arr_desc_id = np.array(efficiencies['desc_id'])
    arr_cold_gas_mass = np.array(efficiencies['cold_gas_mass'])
    arr_hot_gas_mass = np.array(efficiencies['hot_gas_mass'])
    arr_rate_accrete_hot = np.array(efficiencies['rate_accrete_hot'])
    arr_rate_hot_cold = np.array(efficiencies['rate_hot_cold'])
    arr_rate_cold_stars = np.array(efficiencies['rate_cold_stars'])
    arr_rate_cold_hot = np.array(efficiencies['rate_cold_hot'])
    arr_rate_accrete_stars = np.array(efficiencies['rate_accrete_stars'])

    arr_desc_dm_mass = np.array(efficiencies['desc_dm_mass'])
    arr_prog_dm_mass = np.array(efficiencies['prog_dm_mass'])
    arr_diff_dm_mass = np.maximum(arr_desc_dm_mass - arr_prog_dm_mass, 0)
    arr_desc_stellar_mass = np.array(efficiencies['desc_stellar_mass'])

    for i in range(arr_desc_id.shape[0]):
        sub_id = arr_desc_id[i]
        dict_gas_mass[sub_id] = (
            arr_cold_gas_mass[i],
            arr_hot_gas_mass[i],
            arr_rate_accrete_hot[i],
            arr_rate_hot_cold[i],
            arr_rate_cold_stars[i],
            arr_rate_cold_hot[i],
            arr_rate_accrete_stars[i],
            arr_diff_dm_mass[i],
            arr_desc_stellar_mass[i],
            )

    for i, sub_id in enumerate(all_subhalo_ids[:, snap]):
        if sub_id == -1:
            continue
        # TODO: Not sure if setting all to zero is the correct way to deal with skips
        if sub_id not in dict_gas_mass:
            all_trees[i, :, snap] = 0
            continue
        # TODO: Sanity check sum of hot_gas and cold_gas
        cold_gas_mass, hot_gas_mass, rate_accrete_hot, rate_hot_cold, rate_cold_stars, rate_cold_hot, rate_accrete_stars, diff_dm_mass, desc_stellar_mass = dict_gas_mass[sub_id]
        all_trees[i, 2, snap] = cold_gas_mass
        all_trees[i, 4, snap] = hot_gas_mass
        all_trees[i, 9, snap] = rate_accrete_hot
        all_trees[i, 10, snap] = rate_hot_cold
        all_trees[i, 11, snap] = rate_cold_stars
        all_trees[i, 12, snap] = rate_cold_hot
        all_trees[i, 13, snap] = rate_accrete_stars

        all_trees[i, 14, snap] = rate_accrete_hot / diff_dm_mass if diff_dm_mass else -1
        all_trees[i, 15, snap] = rate_hot_cold / hot_gas_mass if hot_gas_mass else -1
        all_trees[i, 16, snap] = rate_cold_stars / cold_gas_mass if cold_gas_mass else -1
        all_trees[i, 17, snap] = rate_cold_hot / rate_cold_stars if rate_cold_stars else -1
        all_trees[i, 18, snap] = rate_accrete_stars / desc_stellar_mass

log(f'Saving data')
save_data_dir = config.get_generated_data_dir()
if not os.path.exists(save_data_dir):
    os.makedirs(save_data_dir)
np.save(f'{save_data_dir}trees.npy', all_trees)
np.save(f'{save_data_dir}tree_ids.npy', all_subhalo_ids)

log(f'Script finished')
