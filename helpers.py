# Lots was lifted from https://github.com/illustristng/illustris_python
import argparse
import os
import datetime
import socket

import numpy as np
import h5py


def log(*message):
    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    print(time, *message)


class Config:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--box_size', type=int, default=100)
        parser.add_argument('--run', type=int, default=3)
        parser.add_argument('--sim', type=str, default='tng')
        parser.add_argument('--snap', type=int, default=99)
        parser, unknown = parser.parse_known_args()
        if unknown:
            log(f'Unknown arguments: {unknown}')
        parser = vars(parser)
        log(f'Config: {parser}')

        self.box_size = parser['box_size']
        self.run = parser['run']
        self.sim = parser['sim']
        self.snap = parser['snap']

        self.name = f'{self.sim}{self.box_size}-{self.run}'
        hostname = socket.gethostname()
        if hostname == 'lenovo-p52':
            self.hostname = 'local'
        elif (hostname == 'cuillin') or ('worker' in hostname) or ('fcfs' in hostname):
            self.hostname = 'cuillin'
        else:
            self.hostname = 'tng'

        if self.sim == 'tng':
            self.hubble_param = 0.6774
            self.omega_m = 0.3089
            self.omega_b = 0.0486
            self.dm_mass_cut = {  # Simulation mass units
                100: {
                    1: 1,
                    3: 100
                },
            }[self.box_size][self.run]
        else:
            raise NotImplementedError

    def get_base_dir(self):
        if self.hostname == 'local':
            return '/home/rmcg/data/'
        elif self.hostname == 'cuillin':
            return '/disk01/rmcg/'
        elif self.hostname == 'tng':
            return '/home/tnguser/'

    def get_generated_data_dir(self):
        data_dir = f'{self.get_base_dir()}generated/sim_sam/{self.name}/'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        return data_dir

    def get_lhalotree_dir(self):
        assert self.hostname != 'tng'
        return f'{self.get_base_dir()}downloaded/tng/{self.name}/merger_tree/lhalotree/'

    def get_gc_dir(self, snap):
        if self.hostname == 'tng':
            assert self.sim == 'tng'
            gc_dir = f'{self.get_base_dir()}sims.TNG/TNG{self.box_size}-{self.run}/'
            gc_dir += f'output/groups_{snap:03d}/'
        else:
            gc_dir = f'{self.get_base_dir()}downloaded/tng/tng{self.box_size}-{self.run}/'
            gc_dir += f'fof_subfind_snapshot_{snap}/'
        return gc_dir

    def gcPath(self, snap, chunkNum=0):
        """ Return absolute path to a group catalog file (modify as needed). """
        filePath1 = self.get_gc_dir(snap) + 'groups_%03d.%d.hdf5' % (snap, chunkNum)
        filePath2 = self.get_gc_dir(snap) + 'fof_subhalo_tab_%03d.%d.hdf5' % (snap, chunkNum)

        if os.path.isfile(filePath1):
            return filePath1
        return filePath2

    def offsetPath(self, snap):
        """ Return absolute path to a separate offset file (modify as needed). """
        if self.hostname == 'tng':
            assert self.sim == 'tng'
            offsetPath = f'{self.get_base_dir()}sims.TNG/TNG{self.box_size}-{self.run}/'
            offsetPath += f'postprocessing/offsets/offsets_{snap:03d}.hdf5'
        else:
            offsetPath = f'{self.get_base_dir()}downloaded/tng/tng{self.box_size}-{self.run}/'
            offsetPath += f'offsets/offsets_{snap:03d}.hdf5'
        return offsetPath

    def snapPath(self, snap, chunkNum=0):
        """ Return absolute path to a snapshot HDF5 file (modify as needed). """
        if self.hostname == 'tng':
            assert self.sim == 'tng'
            snapPath = f'{self.get_base_dir()}sims.TNG/TNG{self.box_size}-{self.run}/'
            snapPath += f'output/snapdir_{snap:03d}/'
        else:
            snapPath = f'{self.get_base_dir()}downloaded/tng/tng{self.box_size}-{self.run}/'
            snapPath += f'snapshot_{snap}/'
        filePath = snapPath + 'snap_' + str(snap).zfill(3)
        filePath += '.' + str(chunkNum) + '.hdf5'
        return filePath

    @staticmethod
    def partTypeNum(partType):
        """ Mapping between common names and numeric particle types. """
        if str(partType).isdigit():
            return int(partType)

        if str(partType).lower() in ['gas','cells']:
            return 0
        if str(partType).lower() in ['dm','darkmatter']:
            return 1
        if str(partType).lower() in ['dmlowres']:
            return 2 # only zoom simulations, not present in full periodic boxes
        if str(partType).lower() in ['tracer','tracers','tracermc','trmc']:
            return 3
        if str(partType).lower() in ['star','stars','stellar']:
            return 4 # only those with GFM_StellarFormationTime>0
        if str(partType).lower() in ['wind']:
            return 4 # only those with GFM_StellarFormationTime<0
        if str(partType).lower() in ['bh','bhs','blackhole','blackholes']:
            return 5

        raise Exception("Unknown particle type name.")

    @staticmethod
    def getNumPart(header):
        """ Calculate number of particles of all types given a snapshot header. """
        nTypes = 6

        nPart = np.zeros(nTypes, dtype=np.int64)
        for j in range(nTypes):
            nPart[j] = header['NumPart_Total'][j] | (header['NumPart_Total_HighWord'][j] << 32)

        return nPart

    def loadSubset(self, snapNum, partType, fields=None, subset=None, mdi=None, sq=True, float32=False):
        """ Load a subset of fields for all particles/cells of a given partType.
            If offset and length specified, load only that subset of the partType.
            If mdi is specified, must be a list of integers of the same length as fields,
            giving for each field the multi-dimensional index (on the second dimension) to load.
              For example, fields=['Coordinates', 'Masses'] and mdi=[1, None] returns a 1D array
              of y-Coordinates only, together with Masses.
            If sq is True, return a numpy array instead of a dict if len(fields)==1.
            If float32 is True, load any float64 datatype arrays directly as float32 (save memory). """
        result = {}

        ptNum = self.partTypeNum(partType)
        gName = "PartType" + str(ptNum)

        # make sure fields is not a single element
        if isinstance(fields, str):
            fields = [fields]

        # load header from first chunk
        with h5py.File(self.snapPath(snapNum), 'r') as f:

            header = dict(f['Header'].attrs.items())
            nPart = self.getNumPart(header)

            # decide global read size, starting file chunk, and starting file chunk offset
            if subset:
                offsetsThisType = subset['offsetType'][ptNum] - subset['snapOffsets'][ptNum, :]

                fileNum = np.max(np.where(offsetsThisType >= 0))
                fileOff = offsetsThisType[fileNum]
                numToRead = subset['lenType'][ptNum]
            else:
                fileNum = 0
                fileOff = 0
                numToRead = nPart[ptNum]

            result['count'] = numToRead

            if not numToRead:
                # print('warning: no particles of requested type, empty return.')
                return result

            # find a chunk with this particle type
            i = 1
            while gName not in f:
                f = h5py.File(self.snapPath(snapNum, i), 'r')
                i += 1

            # if fields not specified, load everything
            if not fields:
                fields = list(f[gName].keys())

            for i, field in enumerate(fields):
                # verify existence
                if field not in f[gName].keys():
                    raise Exception("Particle type ["+str(ptNum)+"] does not have field ["+field+"]")

                # replace local length with global
                shape = list(f[gName][field].shape)
                shape[0] = numToRead

                # multi-dimensional index slice load
                if mdi is not None and mdi[i] is not None:
                    if len(shape) != 2:
                        raise Exception("Read error: mdi requested on non-2D field ["+field+"]")
                    shape = [shape[0]]

                # allocate within return dict
                dtype = f[gName][field].dtype
                if dtype == np.float64 and float32: dtype = np.float32
                result[field] = np.zeros(shape, dtype=dtype)

        # loop over chunks
        wOffset = 0
        origNumToRead = numToRead

        while numToRead:
            f = h5py.File(self.snapPath(snapNum, fileNum), 'r')

            # no particles of requested type in this file chunk?
            if gName not in f:
                f.close()
                fileNum += 1
                fileOff  = 0
                continue

            # set local read length for this file chunk, truncate to be within the local size
            numTypeLocal = f['Header'].attrs['NumPart_ThisFile'][ptNum]

            numToReadLocal = numToRead

            if fileOff + numToReadLocal > numTypeLocal:
                numToReadLocal = numTypeLocal - fileOff

            #print('['+str(fileNum).rjust(3)+'] off='+str(fileOff)+' read ['+str(numToReadLocal)+\
            #      '] of ['+str(numTypeLocal)+'] remaining = '+str(numToRead-numToReadLocal))

            # loop over each requested field for this particle type
            for i, field in enumerate(fields):
                # read data local to the current file
                if mdi is None or mdi[i] is None:
                    result[field][wOffset:wOffset+numToReadLocal] = f[gName][field][fileOff:fileOff+numToReadLocal]
                else:
                    result[field][wOffset:wOffset+numToReadLocal] = f[gName][field][fileOff:fileOff+numToReadLocal, mdi[i]]

            wOffset   += numToReadLocal
            numToRead -= numToReadLocal
            fileNum   += 1
            fileOff    = 0  # start at beginning of all file chunks other than the first

            f.close()

        # verify we read the correct number
        if origNumToRead != wOffset:
            raise Exception("Read ["+str(wOffset)+"] particles, but was expecting ["+str(origNumToRead)+"]")

        # only a single field? then return the array instead of a single item dict
        if sq and len(fields) == 1:
            return result[fields[0]]

        return result

    def getSnapOffsets(self, snapNum, id, type):
        """ Compute offsets within snapshot for a particular group/subgroup. """
        r = {}

        # old or new format
        if 'fof_subhalo' in self.gcPath(snapNum):
            # use separate 'offsets_nnn.hdf5' files
            with h5py.File(self.offsetPath(snapNum), 'r') as f:
                groupFileOffsets = f['FileOffsets/'+type][()]
                r['snapOffsets'] = np.transpose(f['FileOffsets/SnapByType'][()])  # consistency
        else:
            # load groupcat chunk offsets from header of first file
            with h5py.File(self.gcPath(snapNum), 'r') as f:
                groupFileOffsets = f['Header'].attrs['FileOffsets_'+type]
                r['snapOffsets'] = f['Header'].attrs['FileOffsets_Snap']

        # calculate target groups file chunk which contains this id
        groupFileOffsets = int(id) - groupFileOffsets
        fileNum = np.max(np.where(groupFileOffsets >= 0))
        groupOffset = groupFileOffsets[fileNum]

        # load the length (by type) of this group/subgroup from the group catalog
        with h5py.File(self.gcPath(snapNum, fileNum), 'r') as f:
            r['lenType'] = f[type][type+'LenType'][groupOffset, :]

        # old or new format: load the offset (by type) of this group/subgroup within the snapshot
        if 'fof_subhalo' in self.gcPath(snapNum):
            with h5py.File(self.offsetPath(snapNum), 'r') as f:
                r['offsetType'] = f[type+'/SnapByType'][id, :]
        else:
            with h5py.File(self.gcPath(snapNum, fileNum), 'r') as f:
                r['offsetType'] = f['Offsets'][type+'_SnapByType'][groupOffset, :]

        return r

    def loadSubhalo(self, snapNum, id, partType, fields=None):
        """ Load all particles/cells of one type for a specific subhalo
            (optionally restricted to a subset fields). """
        # load subhalo length, compute offset, call loadSubset
        subset = self.getSnapOffsets(snapNum, id, "Subhalo")
        return self.loadSubset(snapNum, partType, fields, subset=subset)

    def loadHalo(self, snapNum, id, partType, fields=None):
        """ Load all particles/cells of one type for a specific halo
            (optionally restricted to a subset fields). """
        # load halo length, compute offset, call loadSubset
        subset = self.getSnapOffsets(snapNum, id, "Group")
        return self.loadSubset(snapNum, partType, fields, subset=subset)

    def get_ages(self):
        assert self.sim == 'tng'
        # Values taken from tng100-1 webpage
        return np.array([0.179, 0.271, 0.37, 0.418, 0.475, 0.517, 0.547, 0.596,
                         0.64, 0.687, 0.732, 0.764, 0.844, 0.932, 0.965, 1.036,
                         1.112, 1.177, 1.282, 1.366, 1.466, 1.54, 1.689, 1.812,
                         1.944, 2.145, 2.238, 2.384, 2.539, 2.685, 2.839, 2.981,
                         3.129, 3.285, 3.447, 3.593, 3.744, 3.902, 4.038, 4.206,
                         4.293, 4.502, 4.657, 4.816, 4.98, 5.115, 5.289, 5.431,
                         5.577, 5.726, 5.878, 6.073, 6.193, 6.356, 6.522, 6.692,
                         6.822, 6.998, 7.132, 7.314, 7.453, 7.642, 7.786, 7.932,
                         8.079, 8.28, 8.432, 8.587, 8.743, 8.902, 9.062, 9.225,
                         9.389, 9.556, 9.724, 9.837, 10.009, 10.182, 10.299, 10.535,
                         10.654, 10.834, 11.016, 11.138, 11.323, 11.509, 11.635, 11.824,
                         11.951, 12.143, 12.337, 12.467, 12.663, 12.795, 12.993, 13.127,
                         13.328, 13.463, 13.667, 13.803])

    def get_redshifts(self):
        assert self.sim == 'tng'
        return np.array([20.05, 14.99, 11.98, 10.98, 10.0, 9.39, 9.0, 8.45, 8.01,
                         7.6, 7.24, 7.01, 6.49, 6.01, 5.85, 5.53, 5.23, 5.0, 4.66,
                         4.43, 4.18, 4.01, 3.71, 3.49, 3.28, 3.01, 2.9, 2.73, 2.58,
                         2.44, 2.32, 2.21, 2.1, 2.0, 1.9, 1.82, 1.74, 1.67, 1.6, 1.53,
                         1.5, 1.41, 1.36, 1.3, 1.25, 1.21, 1.15, 1.11, 1.07, 1.04, 1.0,
                         0.95, 0.92, 0.89, 0.85, 0.82, 0.79, 0.76, 0.73, 0.7, 0.68, 0.64,
                         0.62, 0.6, 0.58, 0.55, 0.52, 0.5, 0.48, 0.46, 0.44, 0.42, 0.4,
                         0.38, 0.36, 0.35, 0.33, 0.31, 0.3, 0.27, 0.26, 0.24, 0.23, 0.21,
                         0.2, 0.18, 0.17, 0.15, 0.14, 0.13, 0.11, 0.1, 0.08, 0.07, 0.06,
                         0.05, 0.03, 0.02, 0.01, 0.0])

    def get_scale_factors(self):
        return 1 / (1 + self.get_redshifts())

    def get_closest_snapshot_for_age(self, age):
        assert self.sim == 'tng'
        min_dist = float('inf')
        closest_snap = 0
        for snap in range(100):
            # Use get, default to -1
            dist = abs(age - self.get_ages()[snap])
            if dist < min_dist:
                min_dist = dist
                closest_snap = snap
        return closest_snap

    @staticmethod
    def get_colors():
        return {
            0: 'g',  # Redshift colors taken from N12
            1: 'r',
            2: 'b',
            3: 'k',
            'dm_mass': 'tab:grey',
            'cold_gas_mass': 'tab:blue',
            'hot_gas_mass': 'tab:red',
            'stellar_mass': 'tab:orange',
        }

    @staticmethod
    def get_names():
        return {
            'f_a': 'Baryon accretion efficiency',
            'f_a_id': 'Baryon accretion efficiency (from particle ids)',
            'f_c': 'Cooling efficiency',
            'f_s': 'Star formation efficiency',
            'f_d': 'Feedback efficiency',
            'f_m': 'Steller accretion efficiency',
            'f_m_id': 'Steller accretion efficiency (from particle ids)',
            'dm_mass': 'Dark matter mass',
            'cold_gas_mass': 'Cold gas mass',
            'hot_gas_mass': 'Hot gas mass',
            'stellar_mass': 'Stellar mass',
        }
