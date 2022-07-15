import os

import h5py
import matplotlib.pyplot as plt
import numpy as np

import helpers

config = helpers.Config()
h = config.get_hubble_param()
gc_dir = config.get_gc_dir(config.snap)

dm_mass = np.array([], dtype='float32')
stellar_mass = np.array([], dtype='float32')
fof_id = np.array([], dtype='int32')
central_id = np.array([], dtype='int32')
for i in range(len(os.listdir(gc_dir))):
    file_name = f'fof_subhalo_tab_{config.snap:03d}.{i}.hdf5'
    print(f'Reading {file_name}')
    with h5py.File(gc_dir+file_name) as file:
        box_size = file['Header'].attrs['BoxSize'] / (1000 * h)

        try:
            mass = np.array(file['Subhalo/SubhaloMassType']) * (10**10) / h
        except KeyError:
            continue
        dm_mass = np.concatenate([dm_mass, mass[:, 1]])
        stellar_mass = np.concatenate([stellar_mass, mass[:, 4]])
        fof_id = np.concatenate([fof_id, np.array(file['Subhalo/SubhaloGrNr'])])
        central_id = np.concatenate([central_id, np.array(file['Group/GroupFirstSub'])])
n_sub = stellar_mass.shape[0]
is_central = np.arange(n_sub) == central_id[fof_id]

mask = (stellar_mass != 0) & (dm_mass != 0)
dm_mass = dm_mass[mask]
stellar_mass = stellar_mass[mask]
is_central = is_central[mask]

stellar_mass = np.log10(stellar_mass)
dm_mass = np.log10(dm_mass)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
ylabel = '$\phi = \mathrm{dn}/\mathrm{dlog}_{10} \mathrm{M}_{*} \,[\mathrm{cMpc}^{-3}]$'

bin_width = .5
bins = np.arange(8, 13, bin_width)
normalisation = bin_width * (box_size**3)
mids = (bins[:-1] + bins[1:]) / 2
n = np.histogram(stellar_mass, bins=bins)[0].astype('float64') / normalisation
axs[0].plot(mids, n, '.-', label='All')
n = np.histogram(stellar_mass[is_central], bins=bins)[0].astype('float64') / normalisation
axs[0].plot(mids, n, '.-', label='Centrals')
axs[0].set_yscale('log')
axs[0].legend(loc='upper right')
axs[0].set_xlabel('$\log_{10}\, M_{*}$  $[\mathrm{M}_\odot]$')
axs[0].set_ylabel(ylabel)

bin_width = .5
bins = np.arange(10, 15, bin_width)
normalisation = bin_width * (box_size**3)
mids = (bins[:-1] + bins[1:]) / 2
n = np.histogram(dm_mass, bins=bins)[0].astype('float64') / normalisation
axs[1].plot(mids, n, '.-', label='All')
n = np.histogram(dm_mass[is_central], bins=bins)[0].astype('float64') / normalisation
axs[1].plot(mids, n, '.-', label='Centrals')
axs[1].set_yscale('log')
axs[1].legend(loc='upper right')
axs[1].set_xlabel('$\log_{10}\, M_{DM}$  $[\mathrm{M}_\odot]$')
axs[1].set_ylabel(ylabel)

plt.tight_layout()
plt.savefig('/home/rmcg/ss_mass_function.png', dpi=300)
plt.close()
