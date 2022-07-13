import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import helpers
from helpers import log

config = helpers.Config()

log(f'Loading pair data')
data = pd.read_parquet(f'{config.get_generated_data_dir()}pairs/snap_{config.snap}.parquet')
n_sub = data.shape[0]
data = {k: np.array(data[k]) for k in data.keys()}

ages = config.get_ages()

print(n_sub)
exit()

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




