#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import socket

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import helpers
from helpers import log


# In[2]:


# TODO: pass these as arguments
snap = 99
sim = 'tng100-3'
log(f'Extracting data for {sim} for snapshot {snap}')

if socket.gethostname() == 'lenovo-p52':
    base_dir = '/home/rmcg/data/'
else:
    base_dir = '/disk01/rmcg/'
sim_dir = f'{base_dir}downloaded/tng/{sim}/'
lhalotree_dir = sim_dir + 'merger_tree/lhalotree/'
with h5py.File(lhalotree_dir+'trees_sf1_099.0.hdf5', 'r') as file:
    redshifts = np.array(file['/Header/Redshifts'])
scale_factors = 1 / (1 + redshifts)
ages = helpers.get_tng_ages()


# ## Finding valid subhalo pairs
# <u> Problems </u>
# - How to define a valid galaxy? Should I implement a stellar mass cut?
# - Assert that diff_dm_mass is positive?

# In[3]:


# desc refers to a subhalo at snap
# prog refers to progenitor subhalo at snap-1

data = {}
data['desc_id'] = np.array([], dtype='int32')
data['prog_id'] = np.array([], dtype='int32')
data['desc_stellar_mass'] = np.array([], dtype='float32')
data['prog_stellar_mass'] = np.array([], dtype='float32')
data['diff_dm_mass'] = np.array([], dtype='float32')
#TODO: run for all files
for file_name in os.listdir(lhalotree_dir)[:1]:
    log(f'Processing {file_name}')
    with h5py.File(lhalotree_dir+file_name, 'r') as file:
        for tree_name in [key for key in file.keys() if 'Tree' in key]:
            # TODO: Units
            arr_sub_id = np.array(file[tree_name+'/SubhaloNumber'])
            arr_gas_mass = np.array(file[tree_name+'/SubhaloMassType'][:, 0])
            arr_dm_mass = np.array(file[tree_name+'/SubhaloMassType'][:, 1])
            arr_stellar_mass = np.array(file[tree_name+'/SubhaloMassType'][:, 4])
            arr_snap_num = np.array(file[tree_name+'/SnapNum'])
            
            arr_central_index = np.array(file[tree_name+'/FirstHaloInFOFGroup'])
            arr_is_central = np.zeros(arr_sub_id.shape[0], dtype=bool)
            for i_sub, i_central in enumerate(arr_central_index):
                arr_is_central[i_sub] = (i_sub == i_central)
                
            arr_prog_index = np.array(file[tree_name+'/FirstProgenitor'])
            arr_prog_id = -1 * np.ones(arr_sub_id.shape[0], dtype='int32')
            arr_prog_dm_mass = np.zeros(arr_sub_id.shape[0], dtype='float32')
            arr_prog_stellar_mass = np.zeros(arr_sub_id.shape[0], dtype='float32')
            for i_desc, i_prog in enumerate(arr_prog_index.copy()):
                if (
                        i_prog != -1 and                      # Check progenitor exists
                        arr_snap_num[i_prog] == snap - 1 and  # Check snapshot isn't skipped
                        arr_is_central[i_prog] and            # Require central subhalo
                        arr_stellar_mass[i_prog] != 0 and     # Require nonzero stellar mass
                        arr_gas_mass[i_prog] != 0             # Require nonzero gas mass
                    ):
                    arr_prog_id[i_desc] = arr_sub_id[i_prog]
                    arr_prog_dm_mass[i_desc] = arr_dm_mass[i_prog]
                    arr_prog_stellar_mass[i_desc] = arr_stellar_mass[i_prog]

            mask = (arr_snap_num == snap)
            mask &= arr_is_central
            mask &= (arr_prog_id != -1)
            mask &= (arr_stellar_mass != 0)
            mask &= (arr_gas_mass != 0)
            
            data['desc_id'] = np.concatenate([data['desc_id'], arr_sub_id[mask]])
            data['prog_id'] = np.concatenate([data['prog_id'], arr_prog_id[mask]])
            data['desc_stellar_mass'] = np.concatenate([data['desc_stellar_mass'], arr_stellar_mass[mask]])
            data['prog_stellar_mass'] = np.concatenate([data['prog_stellar_mass'], arr_prog_stellar_mass[mask]])
            diff_dm_mass = arr_dm_mass[mask] - arr_prog_dm_mass[mask]
            diff_dm_mass[diff_dm_mass < 0] = 0
            data['diff_dm_mass'] = np.concatenate([data['diff_dm_mass'], diff_dm_mass])
            
n_valid = data['desc_id'].shape[0]
log(f'{n_valid} galaxies found')


# <img src="papers/basic_picture.png"/>
# 
# 
# ## Gas rates
# - Use set operations based on particle IDs
# - Calculating gas temp can be taken from [here](https://www.tng-project.org/data/forum/topic/338/cold-and-hot-gas/)
# 
# <u> Problems </u>
# - What happens to particle IDs when gas cells split?
# - Gas cells are different masses in prog and sub snapshots (average over both?)
# - What to return for galaxies which have no cold gas?
# 
# 
# ## Star rates
# - Use GFM_StellarFormationTime to calculate number of stars formed
# - $R_{->star}$ is the rate of stars than were accreted onto the galaxy in the past timestep
# - $f_m = \frac{R_{->star}}{m_{star}}$ (m for merge)
# 
# <u> Problems </u>
# - Stellar particles brought in through mergers aren't accounted for, may mean I underestimate stellar mass
# - Stellar masses evolve over time
# - Wind particles
# - Stellar mass from particles is sometimes more than from subfind (only less by factor of 10^7)

# In[16]:


def calculate_efficiencies(snap, desc_id, prog_id, diff_dm_mass, 
                           desc_stellar_mass, prog_stellar_mass):
    
    # Calculations using gas particles
    gas_fields = ['Masses', 'ParticleIDs', 'StarFormationRate']
    
    prog_g = helpers.loadSubhalo(sim_dir, snap-1, prog_id, 0, fields=gas_fields)
    prog_is_cold_gas = prog_g['StarFormationRate'] > 0
    prog_is_hot_gas = np.logical_not(prog_is_cold_gas)
    prog_cold_gas_ids = prog_g['ParticleIDs'][prog_is_cold_gas]
    prog_hot_gas_ids = prog_g['ParticleIDs'][prog_is_hot_gas]
    
    desc_g = helpers.loadSubhalo(sim_dir, snap, desc_id, 0, fields=gas_fields)
    desc_is_cold_gas = desc_g['StarFormationRate'] > 0
    desc_is_hot_gas = np.logical_not(desc_is_cold_gas)
    desc_cold_gas_ids = desc_g['ParticleIDs'][desc_is_cold_gas]
    desc_hot_gas_ids = desc_g['ParticleIDs'][desc_is_hot_gas]
    
    mean_mass = {}
    for (part_id, mass) in zip(prog_g['ParticleIDs'], prog_g['Masses']):
        mean_mass[part_id] = mass
    for (part_id, mass) in zip(desc_g['ParticleIDs'], desc_g['Masses']):
        if part_id in mean_mass:
            mean_mass[part_id] = (mass + mean_mass[part_id]) / 2
        else:
            mean_mass[part_id] = mass
    
    rate_hot_cold = 0
    hot_to_cold_ids = np.intersect1d(prog_hot_gas_ids, desc_cold_gas_ids)
    for part_id in hot_to_cold_ids:
        rate_hot_cold += mean_mass[part_id]
    rate_hot_cold /= ages[snap] - ages[snap-1]
        
    rate_cold_hot = 0
    cold_to_hot_ids = np.intersect1d(prog_cold_gas_ids, desc_hot_gas_ids)
    for part_id in cold_to_hot_ids:
        rate_cold_hot[part_id] += mean_mass[part_id]
    rate_cold_hot /= ages[snap] - ages[snap-1]
    
    rate_accrete_hot = 0
    accrete_hot_ids = set(np.setdiff1d(desc_hot_gas_ids, prog_g['ParticleIDs']))
    for (part_id, mass) in zip(desc_hot_gas_ids, desc_g['Masses'][desc_is_hot_gas]):
        if part_id in accrete_hot_ids:
            rate_accrete_hot += mass
    rate_accrete_hot /= ages[snap] - ages[snap-1]
    
    cold_gas_mass = np.sum(desc_g['Masses'][desc_is_cold_gas])
    hot_gas_mass = np.sum(desc_g['Masses'][desc_is_hot_gas])
     
    # Calculations using star particles
    stellar_fields = ['GFM_StellarFormationTime', 'Masses', 'ParticleIDs']
    
    desc_s = helpers.loadSubhalo(sim_dir, snap, desc_id, 4, fields=stellar_fields)
    desc_is_star = desc_s['GFM_StellarFormationTime'] > 0  # Remove wind particles
    assert np.isclose(desc_stellar_mass, np.sum(desc_s['Masses'][desc_is_star]), rtol=1e-5)

    prog_s = helpers.loadSubhalo(sim_dir, snap-1, prog_id, 4, fields=stellar_fields)
    prog_is_star = prog_s['GFM_StellarFormationTime'] > 0  # Remove wind particles
    assert np.isclose(prog_stellar_mass, np.sum(prog_s['Masses'][prog_is_star]), rtol=1e-5)
    
    recently_formed = desc_s['GFM_StellarFormationTime'] > scale_factors[snap-1]
    rate_cold_stars = np.sum(desc_s['Masses'][recently_formed])
    rate_cold_stars /= ages[snap] - ages[snap-1]

    rate_accrete_star = 0
    not_accreted_ids = set(np.union1d(desc_s['ParticleIDs'][recently_formed], prog_s['ParticleIDs']))
    for (part_id, mass) in zip(desc_s['ParticleIDs'], desc_s['Masses']):
        if part_id not in not_accreted_ids:
            rate_accrete_star += mass
    rate_accrete_star /= ages[snap] - ages[snap-1]
    
    # Calculating efficiencies
    f_a = rate_accrete_hot / diff_dm_mass if diff_dm_mass else -1
    f_c = rate_hot_cold / hot_gas_mass if hot_gas_mass else -1
    f_s = rate_cold_stars / cold_gas_mass if cold_gas_mass else -1
    f_d = rate_cold_hot / rate_cold_stars if rate_cold_stars else -1
    f_m = rate_accrete_star / desc_stellar_mass
    
    return f_a, f_c, f_s, f_d, f_m


# In[17]:


log(f'Calculating efficiencies')
data['f_a'] = np.zeros(n_valid, dtype='float32')
data['f_c'] = np.zeros(n_valid, dtype='float32')
data['f_s'] = np.zeros(n_valid, dtype='float32')
data['f_d'] = np.zeros(n_valid, dtype='float32')
data['f_m'] = np.zeros(n_valid, dtype='float32')
for i in range(n_valid):
    if not (i+1) % (n_valid // 20):
        log(f'{round(100*(i+1)/n_valid)}% complete')
    
    desc_id = data['desc_id'][i]
    prog_id = data['prog_id'][i]
    diff_dm_mass = data['diff_dm_mass'][i]
    desc_stellar_mass = data['desc_stellar_mass'][i]
    prog_stellar_mass = data['prog_stellar_mass'][i]
    try:
        f_a, f_c, f_s, f_d, f_m = calculate_efficiencies(
            snap, desc_id, prog_id, diff_dm_mass, desc_stellar_mass, prog_stellar_mass
        )
        data['f_a'][i] = f_a
        data['f_c'][i] = f_c
        data['f_s'][i] = f_s
        data['f_d'][i] = f_d
        data['f_m'][i] = f_m
    except AssertionError as error_message:
        log(f'{error_message} AssertionError for i={i}')


# In[6]:


log(f'Saving data')
save_data_dir = f'{base_dir}generated/sim_sam/{sim}/'
if not os.path.exists(save_data_dir):
    os.makedirs(save_data_dir)
pd.DataFrame(data).to_parquet(f'{save_data_dir}snap_{snap}.parquet', index=False)


# In[25]:


# Should this be in a separate file?
for efficiency in ['f_a', 'f_c', 'f_s', 'f_d']:
    fig, ax = plt.subplots(1, dpi=150)
    mask = data[efficiency] != -1
    ax.hist(data[efficiency][mask], density=True, bins=20)
    ax.text(0.5, 0.8, f'Fraction valid: {np.sum(mask)/mask.shape[0]:.2g}',
            transform=ax.transAxes, fontsize=14)
    ax.text(0.5, 0.7, f'Mean: {np.mean(data[efficiency][mask]):.2g}',
            transform=ax.transAxes, fontsize=14)
    ax.set_title(f'{sim.upper()}, z={round(redshifts[snap])}', fontsize=18)
    ax.set_xlabel(f'${efficiency}$', fontsize=14)
    
    plt.savefig(f'/home/rmcg/{sim}_{efficiency}_hist_snap{snap}.png')
    plt.show()
    plt.close()


# In[8]:


# TODO: Multiprocessing
# TODO: Merger tree for single galaxies, print efficiencies (separate file)
# TODO: Save data alongside plots, save plots as pdf
easdgasdkglja


# In[ ]:




