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

max_snap = 99
n_snap = max_snap+1
# TODO: Hot and cold gas metallicity?
gc_properties = ['bh_mass', 'bh_dot', 'dm_mass', 'gas_mass',
                 'gas_metallicity', 'sfr', 'stellar_mass', 'stellar_metallicity']
snap_properties = ['cold_gas_mass', 'hot_gas_mass',
                   'diff_dm_mass', 'rate_accrete_dm', 'rate_accrete_hot', 'rate_accrete_baryon',
                   'rate_hot_cold', 'rate_cold_stars', 'rate_cold_hot', 'rate_accrete_stars', 'rate_merge_stars',
                   'f_a', 'f_a_id', 'f_c', 'f_s', 'f_d', 'f_m', 'f_m_id']


def extract_trees(filepath):
    log(f'Starting processing file: {filepath}')
    with h5py.File(filepath, 'r') as file:
        n_halos_in_tree = np.array(file['/Header/TreeNHalos'])

        file_trees = {prop: np.zeros((0, n_snap), dtype='float64') for prop in gc_properties}
        file_trees['subhalo_id'] = np.zeros((0, n_snap), dtype='int64')

        for i_tree, n_halo in enumerate(n_halos_in_tree):
            arr = {}
            tree = file[f'Tree{i_tree}']

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
            trees = {prop: np.zeros((n_valid_sub_this_file, n_snap), dtype='float64') for prop in gc_properties}
            trees['subhalo_id'] = -1 * np.ones((n_valid_sub_this_file, n_snap), dtype='int64')
            for i_halo in np.arange(n_halo)[valid]:
                i_prog = i_halo
                # Applying same criteria as in extract_pairs
                while (i_prog != -1 and
                       arr['dm_mass'][i_prog] > config.dm_mass_cut and
                       arr['is_central'][i_prog] and
                       arr['gas_mass'][i_prog] != 0 and
                       arr['stellar_mass'][i_prog] != 0):

                    snap_num = arr['snap_num'][i_prog]

                    for prop in gc_properties:
                        trees[prop][i_sub, snap_num] = arr[prop][i_prog]
                    trees['subhalo_id'][i_sub, snap_num] = arr['subhalo_id'][i_prog]

                    i_prog = arr['main_prog_index'][i_prog]

                i_sub += 1

            for prop in gc_properties:
                file_trees[prop] = np.concatenate([file_trees[prop], trees[prop]], axis=0)
            file_trees['subhalo_id'] = np.concatenate([file_trees['subhalo_id'], trees['subhalo_id']], axis=0)

    return file_trees


pool_result = []
pool = multiprocessing.Pool(n_process)
filenames = [lhalotree_dir+name for name in os.listdir(lhalotree_dir)]
while filenames:
    files_to_process, filenames = filenames[:n_process], filenames[n_process:]
    pool_result += pool.map(extract_trees, files_to_process)

all_trees = pool_result.pop(0)
for pool_trees in pool_result:
    for key in all_trees.keys():
        all_trees[key] = np.concatenate([all_trees[key], pool_trees[key]], axis=0)

for prop in snap_properties:
    all_trees[prop] = np.zeros((all_trees['subhalo_id'].shape[0], n_snap), dtype='float64')

efficiencies_data_dir = config.get_generated_data_dir() + 'efficiencies/'
for snap in range(1, 100):
    log(f'Adding efficiencies for snap {snap}')
    efficiencies = pd.read_parquet(f'{efficiencies_data_dir}snap_{snap}.parquet')

    arr_efficiencies = {prop: np.array(efficiencies[prop]) for prop in snap_properties}
    arr_desc_id = np.array(efficiencies['desc_id'])
    arr_stellar_mass = np.array(efficiencies['desc_stellar_mass'])

    dict_efficiencies = {}
    for i in range(arr_desc_id.shape[0]):
        dict_efficiencies[arr_desc_id[i]] = {prop: arr_efficiencies[prop][i] for prop in snap_properties}
        dict_efficiencies[arr_desc_id[i]]['desc_stellar_mass'] = arr_stellar_mass[i]

    for i, sub_id in enumerate(all_trees['subhalo_id'][:, snap]):
        if sub_id == -1:
            continue
        # TODO: Not sure if setting all to zero is the correct way to deal with skips
        if sub_id not in dict_efficiencies:
            for prop in gc_properties:
                all_trees[prop][i, snap] = 0
            all_trees['subhalo_id'][i, snap] = -1
            continue

        assert np.isclose(all_trees['stellar_mass'][i, snap], dict_efficiencies[sub_id]['desc_stellar_mass'])
        for prop in snap_properties:
            all_trees[prop][i, snap] = dict_efficiencies[sub_id][prop]

cut_snap = 32
log(f"Cutting trees that can't be tracked to z={config.get_redshifts()[cut_snap]}")
mask = all_trees['dm_mass'][:, 32] != 0
for k in all_trees:
    all_trees[k] = all_trees[k][mask]

n_trees = np.sum(mask)
all_trees['z'] = np.tile(config.get_redshifts(), (n_trees, 1))

log(f'Saving data')
save_data_dir = config.get_generated_data_dir() + 'trees/'
if not os.path.exists(save_data_dir):
    os.makedirs(save_data_dir)
for prop, arr in all_trees.items():
    np.save(f'{save_data_dir}{prop}.npy', arr)

log(f'Script finished')
