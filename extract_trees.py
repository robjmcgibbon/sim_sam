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
