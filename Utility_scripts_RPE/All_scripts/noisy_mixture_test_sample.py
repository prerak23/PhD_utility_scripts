import os
import numpy as np
import h5py
import soundfile as sf
from scipy import signal
import random
from audiogen.audiogen.noise.synthetic import SSN
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.io.wavfile import write
from scipy.spatial import distance
import math
import acoustics



#Train speech data 100000
#root_speech_data='/srv/storage/talc@talc-data.nancy/multispeech/corpus/speech_recognition/LibriSpeech/train-clean-360/'
#Speech shape noise data 28000
#root_ssn_data='/srv/storage/talc@talc-data.nancy/multispeech/corpus/speech_recognition/LibriSpeech/train-clean-100/'
#Validation data speech
#root_speech_val_data='/srv/storage/talc@talc-data.nancy/multispeech/corpus/speech_recognition/LibriSpeech/dev-clean/'
#Test speech data
#root_speech_test_data='/srv/storage/talc@talc-data.nancy/multispeech/corpus/speech_recognition/LibriSpeech/test-clean/'
#RIR Data
#rir_data_path='/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/generated_rirs_lwh_correct.hdf5'
#Reverb data
#rir_noise_data_path='/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/generated_rirs_lwh_correct_noise.hdf5'

path_speech="/srv/storage/talc@talc-data.nancy/multispeech/corpus/speech_recognition/LibriSpeech/"


speech_files=[]
num=0

#open speech file
#train_file=np.load('speech_files_.npy')
#np.random.shuffle(train_file)

#val_file=np.load('speech_val_files_.npy')
#np.random.shuffle(val_file)

#test_file=np.load('speech_test_files_.npy')
#np.random.shuffle(test_file)

test_file_15sec=np.load("speech_files_15sec.npy")


#open rir file
rir_data=h5py.File("/home/psrivastava/baseline/sofamyroom_2/sample_room_test.hdf5",'r')
#rir_late_reverb_data=h5py.File(rir_noise_data_path,'r')

#Speech shape noise
ssn_no=np.load('ssn_files_.npy')
print(ssn_no)

#Random noise for ref signal
no,sr=sf.read('rand_sig.flac')

#referense signal
ref_sig=h5py.File('gd_rif.hdf5','r')['rif']['room_0']['noise_rir']
ref_sig_ch1=ref_sig[0,:]
ref_sig_ch2=ref_sig[1,:]
filter_ref_sig_ch1=signal.convolve(no,ref_sig_ch1,mode='full')
filter_ref_sig_ch2=signal.convolve(no,ref_sig_ch2,mode='full')
mean_var=np.var(np.concatenate((filter_ref_sig_ch1,filter_ref_sig_ch2),axis=0))

print(mean_var)

#Randomize room
room_nos=[i+1 for i in range(20000)]

#room_nos=[1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4,5,5,5,5,5,6,6,6,6,6]
#room_nos=[207,207,207,207,207]
#Randomize view points
#vp_nos=[1,2,3,4,5]*20000
#vp_nos=[1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5]
vp_nos=[1,2,3,4,5]

#np.random.shuffle(room_nos)
#np.random.shuffle(vp_nos)

#adc=[]
kbc=[]
alc=[]
euc_dist=[]
#rir_enr=[]
#conv_enr=[]
n=[]
sp_var=[]
count_file=0
#Generate 100000 noisy mixtures
#Save file
test_room=np.arange(18000,20000)

xpxs=[[0,1,2,3,4],[5,6,7,8,9],[10,11,12,13,14],[15,16,17,18,19],[20,21,22,23,24],[25,26,27,28,29]]
split_sp=test_file_15sec[68][0].split('-')
dataset=test_file_15sec[68][1]
print(path_speech+dataset+"/"+split_sp[0]+"/"+split_sp[1]+"/"+test_file_15sec[68][0])
sp,srs=sf.read(path_speech+dataset+"/"+split_sp[0]+"/"+split_sp[1]+"/"+test_file_15sec[68][0])
count_file+=1
sp_var.append(-10*np.log(np.var(sp)))

f_=np.empty((30,48000))
adc=[]
for i in xpxs:
    ch_1=rir_data['rirs_sample_test']['room_1']['test_rir_6'][i[0],:]
    ch_2=rir_data['rirs_sample_test']['room_1']['test_rir_6'][i[1],:]
    ch_3=rir_data['rirs_sample_test']['room_1']['test_rir_6'][i[2],:]
    ch_4=rir_data['rirs_sample_test']['room_1']['test_rir_6'][i[3],:]
    ch_5=rir_data['rirs_sample_test']['room_1']['test_rir_6'][i[4],:]
        #print(count_file)

        #Just for the sake of expirement i am taking 3 second signal
    reverb_sig_ch1=signal.convolve(sp,ch_1,mode='full')[20:48020]
    reverb_sig_ch2=signal.convolve(sp,ch_2,mode='full')[20:48020]
    reverb_sig_ch3=signal.convolve(sp,ch_3,mode='full')[20:48020]
    reverb_sig_ch4=signal.convolve(sp,ch_4,mode='full')[20:48020]
    reverb_sig_ch5=signal.convolve(sp,ch_5,mode='full')[20:48020]

        

        #Calculate alpha and beta
    snr_static=random.randint(60,70)
        #snr_diff=random.randint(30,60)

        #sigma_diff=mean_var/(np.power(10,(snr_diff/10)))
    sigma_static=mean_var/(np.power(10,(snr_static/10)))


        #Calculate alpha

        #sigma_ssn=np.var(np.concatenate((laterev_ch1_ssn,laterev_ch2_ssn),axis=0))
        #alpha=np.sqrt((sigma_diff/sigma_ssn))



        #2 Channel White noise
    white_noise_ch1=np.random.normal(0,1,size=48000)
    white_noise_ch2=np.random.normal(0,1,size=48000)
    white_noise_ch3=np.random.normal(0,1,size=48000)
    white_noise_ch4=np.random.normal(0,1,size=48000)
    white_noise_ch5=np.random.normal(0,1,size=48000)

        #Calculate beta
    beta=np.sqrt(sigma_static)

        #fixing alpha and beta
        #alpha_fix=0.001
        #beta_fix=0.001

    static_noise_ch1=white_noise_ch1*beta
    static_noise_ch2=white_noise_ch2*beta
    static_noise_ch3=white_noise_ch3*beta
    static_noise_ch4=white_noise_ch4*beta
    static_noise_ch5=white_noise_ch5*beta

        #static_noise_ch1_fix=white_noise_ch1*beta_fix
        #static_noise_ch2_fix=white_noise_ch2*beta_fix

#diff_noise_ch1=laterev_ch1_ssn*alpha
#diff_noise_ch2=laterev_ch2_ssn*alpha

        #diff_noise_ch1_fix=laterev_ch1_ssn*alpha_fix
        #diff_noise_ch2_fix=laterev_ch2_ssn*alpha_fix

    signalf_ch1=reverb_sig_ch1+static_noise_ch1
    signalf_ch2=reverb_sig_ch2+static_noise_ch2
    signalf_ch3=reverb_sig_ch3+static_noise_ch3
    signalf_ch4=reverb_sig_ch4+static_noise_ch4
    signalf_ch5=reverb_sig_ch5+static_noise_ch5

        #np_save_mix[(vp-1)*2,:]=signalf_ch1
        #np_save_mix[((vp*2)-1),:]=signalf_ch2

        #signalf_ch1=np.add(reverb_sig_ch1,diff_noise_ch1)
        #signalf_ch1=np.add(signalf_ch1,static_noise_ch1)
        #signalf_ch2=np.add(reverb_sig_ch2,diff_noise_ch2)
        #signalf_ch2=np.add(signalf_ch2,static_noise_ch2)

        #Doing this so that we can write in stereo files
        #signalf_ch1_=np.reshape(signalf_ch1,(-1,1))
        #signalf_ch2_=np.reshape(signalf_ch2,(-1,1))
        #abc=np.concatenate((signalf_ch1_,signalf_ch2_),axis=1)

        #diff_noise_f=np.concatenate((diff_noise_ch1,diff_noise_ch2),axis=0)
        #print(diff_noise_f.shape)
    static_noise_f=np.concatenate((static_noise_ch1,static_noise_ch2,static_noise_ch3,static_noise_ch4,static_noise_ch5),axis=0)

        #diff_noise_f_fix=np.concatenate((diff_noise_ch1_fix,diff_noise_ch2_fix),axis=0)
        #static_noise_f_fix=np.concatenate((static_noise_ch1_fix,static_noise_ch2_fix),axis=0)


    snr_f=10*np.log10((np.var(np.concatenate((reverb_sig_ch1,reverb_sig_ch2,reverb_sig_ch3,reverb_sig_ch4,reverb_sig_ch5),axis=0)))/(np.var(static_noise_f)))
        #snr_fix=10*np.log10((np.var(np.concatenate((signalf_ch1_fix,signalf_ch2_fix),axis=0)))/(np.var(np.add(diff_noise_f_fix,static_noise_f_fix))))

        
    adc.append(snr_f)
        #kbc.append(snr_f2)
        #alc.append(snr_f3)
        #kbc.append(snr_fix)
        #rir_enr.append(acoustics.Signal(np.concatenate((ch_1,ch_2),axis=0),16000).energy())
        #conv_enr.append(acoustics.Signal(np.concatenate((reverb_sig_ch1,reverb_sig_ch2),axis=0),16000).energy())

        #print(snr_f)
        #print(snr_f,snr_f2,snr_f3)
        #euc_dist.append(distance.euclidean(rir_data['receiver_config']['room_'+str(room)]['barycenter'][(vp-1),:],rir_data['source_config']['room_'+str(room)]['source_pos'][(vp-1),:]))
        #print("example",i)
        #print(adc)

    f_[i[0],:]=signalf_ch1
    f_[i[1],:]=signalf_ch2
    f_[i[2],:]=signalf_ch3
    f_[i[3],:]=signalf_ch4
    f_[i[4],:]=signalf_ch5
    
    
    print(adc)
    #adc=np.array(adc)
    #print(adc.shape)
    #room_id.create_dataset("nsmix_f",(10,256000),data=np_save_mix)
    #room_id.create_dataset("nsmix_snr_f",(5,),data=adc)
    #room_id.create_dataset("nsmix_snr_diff", 5, data=kbc[-5:])
    #room_id.create_dataset("nsmix_snr_static", 5, data=alc[-5:])

np.save("mix_6_sample_room",f_)

