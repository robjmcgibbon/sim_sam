import os

import h5py
import numpy as np
import pandas as pd

import helpers
from helpers import log

config = helpers.Config()
log(f'Extracting pairs data for {config.name} for snapshot {config.snap}')

lhalotree_dir = config.get_lhalotree_dir()

data = {
    'desc_id': np.array([], dtype='int32'),
    'prog_id': np.array([], dtype='int32'),
    'desc_dm_mass': np.array([], dtype='float32'),
    'prog_dm_mass': np.array([], dtype='float32'),
    'desc_gas_mass': np.array([], dtype='float32'),
    'prog_gas_mass': np.array([], dtype='float32'),
    'desc_stellar_mass': np.array([], dtype='float32'),
    'prog_stellar_mass': np.array([], dtype='float32'),
}
for file_name in os.listdir(lhalotree_dir):
    log(f'Processing {file_name}')
    with h5py.File(lhalotree_dir+file_name, 'r') as file:
        for tree_name in [key for key in file.keys() if 'Tree' in key]:
            arr_sub_id = np.array(file[tree_name+'/SubhaloNumber'])
            arr_mass = np.array(file[tree_name+'/SubhaloMassType'])
            arr_gas_mass = arr_mass[:, 0]
            arr_dm_mass = arr_mass[:, 1]
            arr_stellar_mass = arr_mass[:, 4]
            arr_snap_num = np.array(file[tree_name+'/SnapNum'])
            
            arr_central_index = np.array(file[tree_name+'/FirstHaloInFOFGroup'])
            arr_is_central = np.zeros(arr_sub_id.shape[0], dtype=bool)
            for i_sub, i_central in enumerate(arr_central_index):
                arr_is_central[i_sub] = (i_sub == i_central)

            is_valid_desc = (arr_snap_num == config.snap)
            is_valid_desc &= (arr_dm_mass > config.dm_mass_cut)
            is_valid_desc &= arr_is_central
            is_valid_desc &= (arr_gas_mass != 0)
            is_valid_desc &= (arr_stellar_mass != 0)

            desc_id = arr_sub_id[is_valid_desc]
            desc_dm_mass = arr_dm_mass[is_valid_desc]
            desc_gas_mass = arr_gas_mass[is_valid_desc]
            desc_stellar_mass = arr_stellar_mass[is_valid_desc]

            prog_id = -1 * np.ones(np.sum(is_valid_desc), dtype='int32')
            prog_dm_mass = np.zeros(np.sum(is_valid_desc), dtype='float32')
            prog_gas_mass = np.zeros(np.sum(is_valid_desc), dtype='float32')
            prog_stellar_mass = np.zeros(np.sum(is_valid_desc), dtype='float32')

            arr_prog_index = np.array(file[tree_name+'/FirstProgenitor'])
            for i_desc, i_prog in enumerate(arr_prog_index[is_valid_desc]):
                if (
                        i_prog != -1 and                             # Check progenitor exists
                        arr_snap_num[i_prog] == config.snap - 1 and  # Check snapshot isn't skipped
                        arr_is_central[i_prog] and                   # Require central subhalo
                        arr_gas_mass[i_prog] != 0 and                # Require nonzero gas mass
                        arr_stellar_mass[i_prog] != 0                # Require nonzero stellar mass
                   ):
                    prog_id[i_desc] = arr_sub_id[i_prog]
                    prog_dm_mass[i_desc] = arr_dm_mass[i_prog]
                    prog_stellar_mass[i_desc] = arr_stellar_mass[i_prog]

            has_prog = (prog_id != -1)

            data['desc_id'] = np.concatenate([data['desc_id'], desc_id[has_prog]])
            data['prog_id'] = np.concatenate([data['prog_id'], prog_id[has_prog]])
            data['desc_dm_mass'] = np.concatenate([data['desc_dm_mass'], desc_dm_mass[has_prog]])
            data['prog_dm_mass'] = np.concatenate([data['prog_dm_mass'], prog_dm_mass[has_prog]])
            data['desc_gas_mass'] = np.concatenate([data['desc_gas_mass'], desc_gas_mass[has_prog]])
            data['prog_gas_mass'] = np.concatenate([data['prog_gas_mass'], prog_gas_mass[has_prog]])
            data['desc_stellar_mass'] = np.concatenate([data['desc_stellar_mass'], desc_stellar_mass[has_prog]])
            data['prog_stellar_mass'] = np.concatenate([data['prog_stellar_mass'], prog_stellar_mass[has_prog]])

log(f'Saving data')
save_data_dir = config.get_generated_data_dir() + 'pairs/'
if not os.path.exists(save_data_dir):
    os.makedirs(save_data_dir)
pd.DataFrame(data).to_parquet(f'{save_data_dir}snap_{config.snap}.parquet', index=False)

log(f'Script finished')
