import numpy as np
from torch.utils.data import Dataset,DataLoader
import torch
import h5py
import yaml
from asteroid.filterbanks import STFTFB
from asteroid.filterbanks.enc_dec import Filterbank, Encoder, Decoder
import random 

#abcd=h5py.File('/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/noisy_mixture_lwh_.hdf5','r')
abcd=h5py.File('/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/noisy_mixture_lwh_.hdf5')
rt60_file=h5py.File('/home/psrivastava/baseline/scripts/pre_processing/rt60_anno_room_20_2_median.hdf5','r')
anno_file=h5py.File('/home/psrivastava/baseline/scripts/pre_processing/absorption_surface_calcul.hdf5','r')
test_set=np.load("/home/psrivastava/baseline/scripts/pre_processing/test_random_ar.npy")

class load_data_test():
    def __init__(self,vps):
        self.vps=vps
        

    def return_data(self,room_id):
        random_vps=np.array(random.sample(range(5),self.vps))+1
        sample_col_ch1=np.zeros((1,48000))
        sample_col_ch2=np.zeros((1,48000))
        #rt60_col=np.zeros((1,6))
        print(random_vps) 
        for vp in random_vps:
            bn_sample_vp_ch1=abcd['room_nos']["room_"+str(room_id)]['nsmix_f'][(vp-1)*2,:]
            bn_sample_vp_ch2=abcd['room_nos']["room_"+str(room_id)]['nsmix_f'][((vp*2)-1),:]
            
            #rt60=rt60_file['room_nos']["room_"+str(room_id)]['rt60'][()][(vp-1),:]
            
            sample_col_ch1=np.concatenate((sample_col_ch1,bn_sample_vp_ch1.reshape(1,48000)),axis=0)
            
            sample_col_ch2=np.concatenate((sample_col_ch2,bn_sample_vp_ch2.reshape(1,48000)),axis=0)


            #rt60_col=np.concatenate((rt60_col,rt60.reshape(1,6)),axis=0)

        

        absorp=anno_file['room_nos']["room_"+str(room_id)]['absorption'][()]
        surface=anno_file['room_nos']["room_"+str(room_id)]['surface_area'][0]
        volume=anno_file['room_nos']["room_"+str(room_id)]['volume'][0]
        rt60=rt60_file['room_nos']['room_'+str(room_id)]['rt60'][()].reshape(6)
        return torch.tensor(sample_col_ch1[1:,:]).float(),torch.tensor(sample_col_ch2[1:,:]).float(),torch.tensor(rt60).float(),torch.tensor(absorp).float(),torch.tensor(surface).float(),torch.tensor(volume).float()
            
'''
kd=load_data_test(3)

samp1,samp2,rt,ab,surf,vol=kd.return_data(18001)
print(samp1.shape)
print(rt)
'''



