import os
import socket

import h5py
import matplotlib.pyplot as plt
import numpy as np

import helpers

snaps = [50, 99]
sim = 'tng100-3'

if socket.gethostname() == 'lenovo-p52':
    base_dir = '/home/rmcg/data/'
else:
    base_dir = '/disk01/rmcg/'
sim_dir = f'{base_dir}downloaded/tng/{sim}/'

# Early results indicate that f_a decreases with z
# At higher z halos that merge have lower mass
# From this script low mass halos have a lower baryon fraction

bin_width = .5
bins = np.arange(10, 15, bin_width)
mids = (bins[:-1] + bins[1:]) / 2
n_bins = mids.shape[0]
fig, ax = plt.subplots(1)
for snap in snaps:
    gc_dir = f'{sim_dir}fof_subfind_snapshot_{snap}/'

    bh_mass = np.array([], dtype='float32')
    dm_mass = np.array([], dtype='float32')
    gas_mass = np.array([], dtype='float32')
    stellar_mass = np.array([], dtype='float32')
    fof_id = np.array([], dtype='int32')
    central_id = np.array([], dtype='int32')
    for file_name in sorted(os.listdir(gc_dir)):
        print(file_name)
        with h5py.File(gc_dir+file_name) as file:
            h = file['Header'].attrs['HubbleParam']
            box_size = file['Header'].attrs['BoxSize'] / (1000 * h)
            omega_m = file['Header'].attrs['Omega0']
            omega_b = 0.0486
            z = round(file['Header'].attrs['Redshift'])

            mass = np.array(file['Subhalo/SubhaloMassType']) * (10**10) / h
            bh_mass = np.concatenate([bh_mass, mass[:, 5]])
            dm_mass = np.concatenate([dm_mass, mass[:, 1]])
            gas_mass = np.concatenate([gas_mass, mass[:, 0]])
            stellar_mass = np.concatenate([stellar_mass, mass[:, 4]])
            fof_id = np.concatenate([fof_id, np.array(file['Subhalo/SubhaloGrNr'])])
            central_id = np.concatenate([central_id, np.array(file['Group/GroupFirstSub'])])
    n_sub = stellar_mass.shape[0]
    is_central = np.arange(n_sub) == central_id[fof_id]

    baryon_fraction = (bh_mass + gas_mass + stellar_mass) / dm_mass
    baryon_fraction = baryon_fraction[is_central]
    dm_mass = np.log10(dm_mass[is_central])
    data = []
    for i_bin in range(n_bins):
        mask = (bins[i_bin] < dm_mass) & (dm_mass < bins[i_bin+1])
        data.append(np.mean(baryon_fraction[mask]))
    ax.plot(mids, data, '-', label=f'z={z}')

universal_baryon_fraction = omega_b / (omega_m - omega_b)
ax.axhline(universal_baryon_fraction, ls='--', color='gray', 
        label=r'$\frac{\Omega_b}{\Omega_m-\Omega_b}$')
ax.legend()
plt.show()

