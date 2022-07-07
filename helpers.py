# Lifted from https://github.com/illustristng/illustris_python
import os
import six

import numpy as np
import h5py


def gcPath(basePath, snapNum, chunkNum=0):
    """ Return absolute path to a group catalog file (modify as needed). """
    gcPath = f'{basePath}fof_subfind_snapshot_{snapNum}/'
    filePath1 = gcPath + 'groups_%03d.%d.hdf5' % (snapNum, chunkNum)
    filePath2 = gcPath + 'fof_subhalo_tab_%03d.%d.hdf5' % (snapNum, chunkNum)

    if os.path.isfile(filePath1):
        return filePath1
    return filePath2

def offsetPath(basePath, snapNum):
    """ Return absolute path to a separate offset file (modify as needed). """
    offsetPath = f'{basePath}offsets/offsets_{snapNum:03d}.hdf5'
    return offsetPath

def snapPath(basePath, snapNum, chunkNum=0):
    """ Return absolute path to a snapshot HDF5 file (modify as needed). """
    snapPath = f'{basePath}snapshot_{snapNum}/'
    filePath = snapPath + 'snap_' + str(snapNum).zfill(3)
    filePath += '.' + str(chunkNum) + '.hdf5'
    return filePath


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


def getNumPart(header):
    """ Calculate number of particles of all types given a snapshot header. """
    nTypes = 6

    nPart = np.zeros(nTypes, dtype=np.int64)
    for j in range(nTypes):
        nPart[j] = header['NumPart_Total'][j] | (header['NumPart_Total_HighWord'][j] << 32)

    return nPart


def loadSubset(basePath, snapNum, partType, fields=None, subset=None, mdi=None, sq=True, float32=False):
    """ Load a subset of fields for all particles/cells of a given partType.
        If offset and length specified, load only that subset of the partType.
        If mdi is specified, must be a list of integers of the same length as fields,
        giving for each field the multi-dimensional index (on the second dimension) to load.
          For example, fields=['Coordinates', 'Masses'] and mdi=[1, None] returns a 1D array
          of y-Coordinates only, together with Masses.
        If sq is True, return a numpy array instead of a dict if len(fields)==1.
        If float32 is True, load any float64 datatype arrays directly as float32 (save memory). """
    result = {}

    ptNum = partTypeNum(partType)
    gName = "PartType" + str(ptNum)

    # make sure fields is not a single element
    if isinstance(fields, six.string_types):
        fields = [fields]

    # load header from first chunk
    with h5py.File(snapPath(basePath, snapNum), 'r') as f:

        header = dict(f['Header'].attrs.items())
        nPart = getNumPart(header)

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
            f = h5py.File(snapPath(basePath, snapNum, i), 'r')
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
        f = h5py.File(snapPath(basePath, snapNum, fileNum), 'r')

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


def getSnapOffsets(basePath, snapNum, id, type):
    """ Compute offsets within snapshot for a particular group/subgroup. """
    r = {}

    # old or new format
    if 'fof_subhalo' in gcPath(basePath, snapNum):
        # use separate 'offsets_nnn.hdf5' files
        with h5py.File(offsetPath(basePath, snapNum), 'r') as f:
            groupFileOffsets = f['FileOffsets/'+type][()]
            r['snapOffsets'] = np.transpose(f['FileOffsets/SnapByType'][()])  # consistency
    else:
        # load groupcat chunk offsets from header of first file
        with h5py.File(gcPath(basePath, snapNum), 'r') as f:
            groupFileOffsets = f['Header'].attrs['FileOffsets_'+type]
            r['snapOffsets'] = f['Header'].attrs['FileOffsets_Snap']

    # calculate target groups file chunk which contains this id
    groupFileOffsets = int(id) - groupFileOffsets
    fileNum = np.max(np.where(groupFileOffsets >= 0))
    groupOffset = groupFileOffsets[fileNum]

    # load the length (by type) of this group/subgroup from the group catalog
    with h5py.File(gcPath(basePath, snapNum, fileNum), 'r') as f:
        r['lenType'] = f[type][type+'LenType'][groupOffset, :]

    # old or new format: load the offset (by type) of this group/subgroup within the snapshot
    if 'fof_subhalo' in gcPath(basePath, snapNum):
        with h5py.File(offsetPath(basePath, snapNum), 'r') as f:
            r['offsetType'] = f[type+'/SnapByType'][id, :]
    else:
        with h5py.File(gcPath(basePath, snapNum, fileNum), 'r') as f:
            r['offsetType'] = f['Offsets'][type+'_SnapByType'][groupOffset, :]

    return r


def loadSubhalo(basePath, snapNum, id, partType, fields=None):
    """ Load all particles/cells of one type for a specific subhalo
        (optionally restricted to a subset fields). """
    # load subhalo length, compute offset, call loadSubset
    subset = getSnapOffsets(basePath, snapNum, id, "Subhalo")
    return loadSubset(basePath, snapNum, partType, fields, subset=subset)


def loadHalo(basePath, snapNum, id, partType, fields=None):
    """ Load all particles/cells of one type for a specific halo
        (optionally restricted to a subset fields). """
    # load halo length, compute offset, call loadSubset
    subset = getSnapOffsets(basePath, snapNum, id, "Group")
    return loadSubset(basePath, snapNum, partType, fields, subset=subset)


def get_tng_ages():
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
