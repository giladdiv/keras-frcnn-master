import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects

    from: `StackOverflow <http://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries>`_
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict


def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if strg =='objects':
            if type(elem) is not np.ndarray:
                dict[strg+'{}'.format(0)] = _todict(elem)
                dict['objects_num'] = 1
            else:
                for ii in range((len(elem))):
                    dict[strg+'{}'.format(ii)] = _todict(elem[ii])
                dict['objects_num'] = ii+1
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            if type(elem) is not unicode:
                dict[strg] = elem
            else:
                dict[strg] = elem.encode('ascii','ignore')
    return dict

if __name__ == "__main__":
    matfile = '/home/gilad/ssd/frcnn/VOCdevkit/VOC3D/Annotations/aeroplane_pascal/2008_000251.mat'
    matdata = loadmat(matfile)['record']
    print('here')

