import numpy as np
import pickle
import matplotlib.pyplot as plt

mode = 'train'
cls_name = 'aeroplane'

if mode =='test':
    with open('../test_dist_{}.pickle'.format(cls_name)) as f:
        az = pickle.load(f)
        air =  np.bincount(az[0])
        air = np.pad(air,(0,360-len(air)),'constant')
        plt.bar(range(0,360,1),air)
        plt.title('Test set {}'.format(cls_name))
        plt.show()
else:
    with open('azimuth_distribution.pickle') as f:
        az = pickle.load(f)
        air =  np.bincount(az[0][cls_name])
        air = np.pad(air,(0,360-len(air)),'constant')
        plt.bar(range(0,360,1),air)
        plt.title('Train set {}'.format(cls_name))
        plt.show()