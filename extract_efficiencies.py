import os

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
scale_factors = config.get_scale_factors()


def calculate_efficiencies(snap, desc_id, prog_id, diff_dm_mass,
                           desc_stellar_mass, prog_stellar_mass):
    
    # Calculations using gas particles
    gas_fields = ['Masses', 'ParticleIDs', 'StarFormationRate']

    prog_g = config.loadSubhalo(snap-1, prog_id, 0, fields=gas_fields)
    prog_is_cold_gas = prog_g['StarFormationRate'] > 0
    prog_is_hot_gas = np.logical_not(prog_is_cold_gas)
    prog_cold_gas_ids = prog_g['ParticleIDs'][prog_is_cold_gas]
    prog_hot_gas_ids = prog_g['ParticleIDs'][prog_is_hot_gas]
    
    desc_g = config.loadSubhalo(snap, desc_id, 0, fields=gas_fields)
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
        rate_cold_hot += mean_mass[part_id]
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
    stellar_fields = ['GFM_InitialMass', 'GFM_StellarFormationTime', 'Masses', 'ParticleIDs']

    prog_s = config.loadSubhalo(snap-1, prog_id, 4, fields=stellar_fields)
    prog_is_star = prog_s['GFM_StellarFormationTime'] > 0  # Remove wind particles
    assert np.isclose(prog_stellar_mass, np.sum(prog_s['Masses'][prog_is_star]), rtol=1e-5)

    desc_s = config.loadSubhalo(snap, desc_id, 4, fields=stellar_fields)
    desc_is_star = desc_s['GFM_StellarFormationTime'] > 0  # Remove wind particles
    assert np.isclose(desc_stellar_mass, np.sum(desc_s['Masses'][desc_is_star]), rtol=1e-5)

    recently_formed = desc_s['GFM_StellarFormationTime'] > scale_factors[snap-1]
    rate_cold_stars = np.sum(desc_s['GFM_InitialMass'][recently_formed])
    rate_cold_stars /= ages[snap] - ages[snap-1]

    rate_accrete_stars = 0
    not_accreted_ids = set(np.union1d(desc_s['ParticleIDs'][recently_formed], prog_s['ParticleIDs']))
    for (part_id, mass) in zip(desc_s['ParticleIDs'], desc_s['Masses']):
        if part_id not in not_accreted_ids:
            rate_accrete_stars += mass
    rate_accrete_stars /= ages[snap] - ages[snap-1]
    
    # Calculating efficiencies
    # f_a = rate_accrete_hot / diff_dm_mass if diff_dm_mass else -1
    # f_c = rate_hot_cold / hot_gas_mass if hot_gas_mass else -1
    # f_s = rate_cold_stars / cold_gas_mass if cold_gas_mass else -1
    # f_d = rate_cold_hot / rate_cold_stars if rate_cold_stars else -1
    # f_m = rate_accrete_star / desc_stellar_mass

    # return f_a, f_c, f_s, f_d, f_m
    return hot_gas_mass, cold_gas_mass, rate_accrete_hot, rate_hot_cold, rate_cold_stars, rate_cold_hot, rate_accrete_stars


log(f'Calculating efficiencies')
data['hot_gas_mass'] = np.zeros(n_sub, dtype='float32')
data['cold_gas_mass'] = np.zeros(n_sub, dtype='float32')
data['rate_accrete_hot'] = np.zeros(n_sub, dtype='float32')
data['rate_hot_cold'] = np.zeros(n_sub, dtype='float32')
data['rate_cold_stars'] = np.zeros(n_sub, dtype='float32')
data['rate_cold_hot'] = np.zeros(n_sub, dtype='float32')
data['rate_accrete_stars'] = np.zeros(n_sub, dtype='float32')
# data['f_a'] = np.zeros(n_sub, dtype='float32')
# data['f_c'] = np.zeros(n_sub, dtype='float32')
# data['f_s'] = np.zeros(n_sub, dtype='float32')
# data['f_d'] = np.zeros(n_sub, dtype='float32')
# data['f_m'] = np.zeros(n_sub, dtype='float32')
for i in range(n_sub):
    if (n_sub // 20) and not (i+1) % (n_sub // 20):
        log(f'{round(100*(i+1)/n_sub)}% complete')
    
    desc_id = data['desc_id'][i]
    prog_id = data['prog_id'][i]
    desc_stellar_mass = data['desc_stellar_mass'][i]
    prog_stellar_mass = data['prog_stellar_mass'][i]
    diff_dm_mass = max(data['desc_dm_mass'][i] - data['prog_dm_mass'][i], 0)
    hot_gas_mass, cold_gas_mass, rate_accrete_hot,rate_hot_cold, rate_cold_stars, rate_cold_hot, rate_accrete_stars = calculate_efficiencies(
        config.snap, desc_id, prog_id, diff_dm_mass, desc_stellar_mass, prog_stellar_mass
    )
    data['hot_gas_mass'][i] = hot_gas_mass
    data['cold_gas_mass'][i] = cold_gas_mass
    data['rate_accrete_hot'][i] = rate_accrete_hot
    data['rate_hot_cold'][i] = rate_hot_cold
    data['rate_cold_stars'][i] = rate_cold_stars
    data['rate_cold_hot'][i] = rate_cold_hot
    data['rate_accrete_stars'][i] = rate_accrete_stars
log(f'Saving data')
save_data_dir = config.get_generated_data_dir() + 'efficiencies/'
if not os.path.exists(save_data_dir):
    os.makedirs(save_data_dir)
pd.DataFrame(data).to_parquet(f'{save_data_dir}snap_{config.snap}.parquet', index=False)

log(f'Script finished')
