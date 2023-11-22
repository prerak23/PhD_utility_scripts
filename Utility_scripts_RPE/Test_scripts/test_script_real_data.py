import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
import h5py
import data_loader_test
#import mlh_baseline as net
import microsoft_vol_exp as net
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gammatone.gtgram as gt
from scipy.signal import butter, lfilter



sns.set_theme()



device=torch.device("cpu")
model_1=net.Model_1().to(device="cpu")
#model_2=net.Model_2().to(device="cuda")
#model_3=net.Model_3().to(device="cpu")
#model_4=net.Ensemble(model_1,model_3,1).to(device="cpu")
#model_4.batch_size=1

#chkp=torch.load("/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/results_mono_exp/mlhm_tas_save_best_sh.pt",map_location=device)
chkp=torch.load("/home/psrivastava/baseline/scripts/pre_processing/results_microsoft_vol_exp/microsoft_volume_tas_save_best_sh.pt",map_location=device)


#chkp=torch.load("/home/psrivastava/baseline/scripts/pre_processing/results_bn_exp/mlh_tas_save_best_sh.pt",map_location=device)
#chkp=torch.load("/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/bn_mlh_/bn_mlh_k2mlh_tas_save_best_sh.pt",map_location=device)

print(chkp["epoch"])
#print(chkp['model_dict_2'])
model_1.load_state_dict(chkp['model_dict']) #model_dict_1
#model_2.load_state_dict(chkp['model_dict_2'])
#model_3.load_state_dict(chkp['model_dict_3'])
#model_4.load_state_dict(chkp['model_dict_ens'])
optimizer = optim.Adam(model_1.parameters(), lr=0.0001) #model_4
optimizer.load_state_dict(chkp['optimizer_dic'])

model_1.eval()
#model_2.eval()
#model_3.eval()
#model_4.eval()

abc=np.load("/home/psrivastava/baseline/scripts/pre_processing/test_random_ar.npy")

no_of_vps=1

test_data=data_loader_test.load_data_test(no_of_vps)

rt60={'000000':[0,0,0.18,0.14,0.16,0.22],'011000':[0,0,0.4,0.33,0.25,0.25],'011100':[0,0,0.46,0.34,0.3,0.37],'011110':[0,0,0.75,0.73,0.68,0.81],'010000':[0,0,0.22,0.19,0.18,0.22],'001000':[0,0,0.22,0.19,0.18,0.22],'000100':[0,0,0.21,0.19,0.2,0.23],'000010':[0,0,0.21,0.18,0.18,0.26],'000001':[0,0,0.22,0.19,0.18,0.24]}
ab={'000000':[0,0,0.42,0.52,0.5,0.37],'011000':[0,0,0.23,0.28,0.34,0.35]
,'011100':[0,0,0.2,0.25,0.3,0.29],'011110':[0,0,0.13,0.13,0.14,0.12]
,'010000':[0,0,0.39,0.44,0.44,0.38],'001000':[0,0,0.39,0.44,0.44,0.38],
'000100':[0,0,0.38,0.41,0.42,0.33],'000010':[0,0,0.4,0.44,0.44,0.32],
'000001':[0,0,0.35,0.43,0.44,0.34]}

volr=[80.141]
sa=[123.026]


no_of_rooms=np.arange(18001,20000)


#no_of_rooms=np.load("/home/psrivastava/baseline/scripts/pre_processing/robustness/high_reverb_rooms.npy")

def lp_filter(ch1_,fs,cutoff,order):
    nyq=0.5*fs
    normal_cutoff=cutoff/nyq
    b,a=butter(order,normal_cutoff,btype="low",analog=False)
    y=lfilter(b,a,ch1_)
    #print("low-pass filter",y.shape)
    return y[:1499]
    
def cal_features(ch1):
    
    feat_gt_=gt.gtgram(ch1,16000,0.004,0.002,58,20)[:20,:]
    feat_dft=np.abs(np.fft.fft(ch1,n=48000)[:1499])
    sort_feat_dft=np.sort(feat_dft)
    lp=lp_filter(ch1,16000,500,10)
    cepstrum=np.fft.ifft(np.log(np.abs(np.fft.fft(ch1,n=48000))))[:1499]
    #print(lp)
    tot_=np.concatenate((feat_gt_,feat_dft.reshape(1,1499),sort_feat_dft.reshape(1,1499),lp.reshape(1,1499),cepstrum.reshape(1,1499)),axis=0)
    #print(tot_)
    #print(tot_.shape)
    print(tot_.shape)

    return tot_ 
        

'''
def cal_features(ch_1,ch_2):
    bz=no_of_vps

    enc_ch1 = torch.stft(ch_1.view(bz, -1), n_fft=1536, hop_length=768, return_complex=True)

    enc_ch2 = torch.stft(ch_2.view(bz, -1), n_fft=1536, hop_length=768, return_complex=True)

    f = torch.view_as_real(enc_ch1)

    f = torch.sqrt(f[:, :, :, 0] ** 2 + f[:, :, :, 1] ** 2)  # Magnitude


    #f2 = torch.view_as_real(enc_ch2)

    #f2 = torch.sqrt(f2[:, :, :, 0] ** 2 + f2[:, :, :, 1] ** 2)  # Magnitude

 
    # Ipd ild calculation
    
    cc = enc_ch1 * torch.conj(enc_ch2)
    ipd = cc / (torch.abs(cc)+10e-8)
    ipd_ri = torch.view_as_real(ipd)
    ild = torch.log(torch.abs(enc_ch1) + 10e-8) - torch.log(torch.abs(enc_ch2) + 10e-8)

    x2 = torch.cat((ipd_ri[:, :, :, 0], ipd_ri[:, :, :, 1], ild), axis=1)

    print(f.shape, x2.shape)
    

    return f ,x2

'''
estimate=np.zeros((1,2)) #28
m=nn.AvgPool1d(32,stride=1)
#o=nn.AvgPool1d(54,stride=1)
n=nn.AvgPool1d(2,stride=2)

#rt60_hist=np.zeros((1,6))
#ab_hist=np.zeros((1,6))
#suf_vol=np.zeros((1,2))

path="/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/diego_real_wet_speech/"
files_=os.listdir(path)
#itr=[[(1,5),(6,10),(11,15),(16,20),(21,25)],[(6,10),(11,15),(16,20),(21,25),(26,30)],[(1,5),(6,10),(16,20),(21,25),(26,30)],[(1,5),(6,10),(11,15),(21,25),(26,30)],[(1,5),(6,10),(11,15),(16,20),(26,30)]]
itr=[(1,5),(6,10),(11,15),(16,20),(21,25)]

for r in files_:
    room_wet_speech=np.load(path+r)
    room_no=r.split("-")[-1].split(".")[0]
    f_=np.concatenate((np.array([sa]),np.array([volr]),np.array([rt60[room_no]]).reshape(1,6),np.array([ab[room_no]]).reshape(1,6)),axis=1)
    estimated=np.zeros((1,2))
    print(room_no)
    for s in [0,1,2,3,4,5]:
        
        total_num=np.zeros((1,14))
        total_precision=np.zeros((1,14))
        total_pred=np.zeros((1,14))
        for mic in itr:    
            combined_ch_1=room_wet_speech[48000:96000,mic[0]:mic[1]:3,s]  #mic[0],mic[1]  #32000,80000
            #combined_ch_2=room_wet_speech[48000:96000,mic[1][0]:mic[1][1]:3,s]  #mic[0],mic[1]
            #combined_ch_3=room_wet_speech[48000:96000,mic[2][0]:mic[2][1]:3,s]  #mic[0],mic[1]
            #combined_ch_4=room_wet_speech[48000:96000,mic[3][0]:mic[3][1]:3,s]  #mic[0],mic[1]
            #combined_ch_5=room_wet_speech[48000:96000,mic[4][0]:mic[4][1]:3,s]  #mic[0],mic[1]
        
            combined_ch_1=torch.tensor(combined_ch_1).float()
            #combined_ch_2=torch.tensor(combined_ch_2).float()
            #combined_ch_3=torch.tensor(combined_ch_3).float()
            #combined_ch_4=torch.tensor(combined_ch_4).float()
            #combined_ch_5=torch.tensor(combined_ch_5).float()
        

            #ch1_feat_1,ch2_feat_1=cal_features(combined_ch_1[:,0],combined_ch_1[:,1])
            #ch1_feat_1,ch2_feat_1=cal_features(combined_ch_1[:,0],combined_ch_1[:,1])
            #mic_x=torch.tensor(combined_ch_1[:,0]).float()
            mic_x=cal_features(combined_ch_1[:,0].reshape(48000))
            #ch1_feat_2,ch2_feat_2=cal_features(combined_ch_2[:,0],combined_ch_2[:,1])
            #ch1_feat_3,ch2_feat_3=cal_features(combined_ch_3[:,0],combined_ch_3[:,1])
            #ch1_feat_4,ch2_feat_4=cal_features(combined_ch_4[:,0],combined_ch_4[:,1])
            #ch1_feat_5,ch2_feat_5=cal_features(combined_ch_5[:,0],combined_ch_5[:,1])
              
            #ch1_feat=torch.tensor(ch1_feat).float().to(device="cpu")
            #ch2_feat=torch.tensor(ch2_feat).float().to(device="cpu")

            #x=model_1(mic_x.reshape(1,769,63))#ch2_feat_1.reshape(1,2307,63))

            x=model_1(torch.tensor(mic_x).unsqueeze(0).float())
            #x=model_1(ch1_feat_1.reshape(1,769,63))
            #print(x.shape)
            #x_2=model_1(ch1_feat_2.reshape(1,769,63))#ch2_feat_2.reshape(1,2307,63))
            #x_3=model_1(ch1_feat_3.reshape(1,769,63))#ch2_feat_3.reshape(1,2307,63))
            #x_4=model_1(ch1_feat_4.reshape(1,769,63))#ch2_feat_4.reshape(1,2307,63))
            #x_5=model_1(ch1_feat_5.reshape(1,769,63))#ch2_feat_5.reshape(1,2307,63))
        




            #x=torch.cat([x,x2],dim=1)
            #x_2=torch.cat([x_2,x_2_],dim=1)
            #x_3=torch.cat([x_3,x_3_],dim=1)
            #x_4=torch.cat([x_4,x_4_],dim=1)
            #x_5=torch.cat([x_5,x_5_],dim=1)
            



 
            #x=m(x)
            #x_2=m(x_2)
            #x_3=m(x_3)
            #x_4=m(x_4)
            #x_5=m(x_5)




            
            #mean,variance=model_3(x.reshape(1,96))
            '''
            mean_2,variance_2=model_3(x_2.reshape(1,96))
            mean_3,variance_3=model_3(x_3.reshape(1,96))
            mean_4,variance_4=model_3(x_4.reshape(1,96))
            mean_5,variance_5=model_3(x_5.reshape(1,96))


            precision_1=1/variance
            precision_2=1/variance_2
            precision_3=1/variance_3
            precision_4=1/variance_4
            precision_5=1/variance_5
       
            total_num=total_num+(mean.detach().numpy()*precision_1.detach().numpy())
            total_num=total_num+(mean_2.detach().numpy()*precision_2.detach().numpy())
            total_num=total_num+(mean_3.detach().numpy()*precision_3.detach().numpy())
            total_num=total_num+(mean_4.detach().numpy()*precision_4.detach().numpy())
            total_num=total_num+(mean_5.detach().numpy()*precision_5.detach().numpy())
    




            total_precision=total_precision+precision_1.detach().numpy()
            total_precision=total_precision+precision_2.detach().numpy()
            total_precision=total_precision+precision_3.detach().numpy()
            total_precision=total_precision+precision_4.detach().numpy()
            total_precision=total_precision+precision_5.detach().numpy()
    



    

            final_mean=total_num/total_precision
    
            final_precision=1/total_precision
            
            #total_pred=np.concatenate((total_pred,final_mean.reshape(1,14)),axis=0)
            '''
            

            tmp=np.concatenate((x.detach().numpy(),np.array([volr]).reshape(1,1)),axis=1) #final_mean
            estimated=np.concatenate((estimated,tmp),axis=0)

    np.save("micro_real_data_"+room_no+".npy",estimated)


            

















'''

for r in no_of_rooms:
    
    print('room no',r)
    bnf_mixture_ch1,bnf_mixture_ch2,rt60,ab,surf,vol=test_data.return_data(r)
    
    #rt60_hist=np.concatenate((rt60_hist,rt60.reshape(1,6)),axis=0)
    #ab_hist=np.concatenate((ab_hist,ab.reshape(1,6)),axis=0)

    #s_v=np.array([surf,vol]).reshape(1,2)
    #suf_vol=np.concatenate((suf_vol,s_v),axis=0)


    #For 15 sec signals
    #first_ex=bnf_mixture_ch1[0,:48000]
    #second_ex=bnf_mixture_ch1[0,48000:96000]
    #third_ex=bnf_mixture_ch1[0,96000:144000]
    #fourth_ex=bnf_mixture_ch1[0,144000:192000]
    #fifth_ex=bnf_mixture_ch1[0,192000:240000]
    
    #first_ex_ch2=bnf_mixture_ch1[0,:48000]
    #second_ex_ch2=bnf_mixture_ch1[0,48000:96000]
    #third_ex_ch2=bnf_mixture_ch1[0,96000:144000]
    #fourth_ex_ch2=bnf_mixture_ch1[0,144000:192000]
    #fifth_ex_ch2=bnf_mixture_ch1[0,192000:240000]
    
    #ch1_feat_first_ex,ch2_feat_first_ex=cal_features(first_ex,first_ex_ch2)
    #ch1_feat_second_ex,ch2_feat_second_ex=cal_features(second_ex,second_ex_ch2)



    #ch1_feat_third_ex,ch2_feat_third_ex=cal_features(third_ex,third_ex_ch2)
    #ch1_feat_fourth_ex,ch2_feat_fourth_ex=cal_features(fourth_ex,fourth_ex_ch2)
    #ch1_feat_fifth_ex,ch2_feat_fifth_ex=cal_features(fifth_ex,fifth_ex_ch2)

    




    ch1_feat,ch2_feat=cal_features(bnf_mixture_ch1,bnf_mixture_ch2)
    
    #print("vp feat ch1",ch1_feat.shape)
    
    #print("vp feat ch2",ch2_feat.shape)


    total_mean_local=np.zeros((1,14))
    
    total_variance_local=np.zeros((1,14))
    
    total_precision=np.zeros((1,14))
    
    total_num=np.zeros((1,14))

    f=np.concatenate((surf.reshape(1,1),vol.reshape(1,1),rt60.reshape(1,6),ab.reshape(1,6)),axis=1)
    
    
    for vp in range(no_of_vps):

        feat_1=ch1_feat[vp,:,:].clone().requires_grad_(True).to(device="cpu")
        feat_2=ch2_feat[vp,:,:].clone().requires_grad_(True).to(device="cpu")
        #print(feat_1)
        #print(feat_2)
        feat_1=torch.unsqueeze(feat_1,axis=0)
        feat_2=torch.unsqueeze(feat_2,axis=0)
        
        #print(model_1(feat_1).shape)
        #print(model_2(feat_2).shape)
        #print(model_2(torch.randn((1,2307,63)).float().to(device="cuda")))
        
        #x1=feat_1.reshape(1,769,63)
        
        #x2=feat_2.reshape(1,2307,63)
        
    

    
    #print(x1.shape)
    #print(x2.shape)

    x,x2=model_1(ch1_feat[0,:,:].reshape(1,769,63),ch2_feat[0,:,:].reshape(1,2307,63))
    x_2,x2_2=model_1(ch1_feat[1,:,:].reshape(1,769,63),ch2_feat[1,:,:].reshape(1,2307,63))
    x_3,x2_3=model_1(ch1_feat[2,:,:].reshape(1,769,63),ch2_feat[2,:,:].reshape(1,2307,63))
    x_4,x2_4=model_1(ch1_feat[3,:,:].reshape(1,769,63),ch2_feat[3,:,:].reshape(1,2307,63))
    x_5,x2_5=model_1(ch1_feat[4,:,:].reshape(1,769,63),ch2_feat[4,:,:].reshape(1,2307,63))
    
    #x1,x2=model_1(ch1_feat[0,:,:].reshape(1,769,63),ch2_feat[0,:,:].reshape(1,2307,63))

    #print(x.shape)
    #x2=model_1(ch1_feat_second_ex[0,:,:].reshape(1,769,63)) #1
    #x3=model_1(ch1_feat_third_ex[0,:,:].reshape(1,769,63))  #2
    #x4=model_1(ch1_feat_fourth_ex[0,:,:].reshape(1,769,63)) #3
    #x5=model_1(ch1_feat_fifth_ex[0,:,:].reshape(1,769,63))  #4
    


    #x=torch.cat((x,x2),axis=1)
    #print(x.shape)
    #x=n(x.reshape(1,32,-1))
    #print(x.shape)

    
    #x_con=torch.cat([x,x2],dim=-1).view(-1,x.shape[-1])
    #x=n(x_con.reshape(1,32,192))
    
    #x=torch.cat([x1,x2],dim=1)
    #x=m(x.reshape(1,1248,32))
    #x2=o(x2.reshape(1,1152,54))
    #x2=torch.cat([x,x2],dim=1)

    #print(x.shape)
    #x=x.reshape(1,-1)
    #mean,variance=model_3(x.reshape(1,1248))
    #mean,variance=model_3(x2.reshape(1,1248))

    
    #x=m(x)
    #x_2=m(x_2)
    #x_3=m(x_3)
    #x_4=m(x_4)
    #x_5=m(x_5)
    #x2=o(x2)
    #x2_2=o(x2_2)
    #x2_3=o(x2_3)
    #x2_4=o(x2_4)
    #x2_5=o(x2_5)
    
    x=torch.cat([x,x2],dim=1)
    x2=torch.cat([x_2,x2_2],dim=1)
    x3=torch.cat([x_3,x2_3],dim=1)
    x4=torch.cat([x_4,x2_4],dim=1)
    x5=torch.cat([x_5,x2_5],dim=1)
    
    x=m(x)
    x2=m(x2)
    x3=m(x3)
    x4=m(x4)
    x5=m(x5)
 




    mean_1,variance_1=model_3(x.reshape(1,1248))
    mean_2,variance_2=model_3(x2.reshape(1,1248))
    mean_3,variance_3=model_3(x3.reshape(1,1248))
    mean_4,variance_4=model_3(x4.reshape(1,1248))
    mean_5,variance_5=model_3(x5.reshape(1,1248))
    



    
    x=m(x)
    x2=m(x2)
    x3=m(x3)
    x4=m(x4)
    x5=m(x5)

    mean_1,variance_1=model_3(x.reshape(1,96))
    mean_2,variance_2=model_3(x2.reshape(1,96))
    mean_3,variance_3=model_3(x3.reshape(1,96))
    mean_4,variance_4=model_3(x4.reshape(1,96))
    mean_5,variance_5=model_3(x5.reshape(1,96))
    



    #mean=[] 
    
    #print(mean_1.shape)
    
    #print(mean_1[0,0])
    
    
    #print(mean_2[0,0])

    #Average the final estimate
    

    #x=x.reshape(1,-1)
    
    #mean,variance=model_4(feat_1,feat_2)

    #print(mean)
    #print(variance)
    #total_mean_local=np.concatenate((total_mean_local,mean.detach().numpy()),axis=0)
        
    #total_variance_local=np.concatenate((total_variance_local,variance.detach().numpy()),axis=0)
        
    #total_num=total_num+(mean.detach().numpy()*precision.detach().numpy())

    #final_precision=1/total_precision
    #mean=np.array(mean).reshape(1,14)
    
    
    total_mean_local=np.concatenate((total_mean_local,mean_1.detach().numpy()),axis=0)
    total_mean_local=np.concatenate((total_mean_local,mean_2.detach().numpy()),axis=0)
    total_mean_local=np.concatenate((total_mean_local,mean_3.detach().numpy()),axis=0)
    total_mean_local=np.concatenate((total_mean_local,mean_4.detach().numpy()),axis=0)
    total_mean_local=np.concatenate((total_mean_local,mean_5.detach().numpy()),axis=0)
    
        
    total_variance_local=np.concatenate((total_variance_local,variance_1.detach().numpy()),axis=0)
    total_variance_local=np.concatenate((total_variance_local,variance_2.detach().numpy()),axis=0)
    total_variance_local=np.concatenate((total_variance_local,variance_3.detach().numpy()),axis=0)
    total_variance_local=np.concatenate((total_variance_local,variance_4.detach().numpy()),axis=0)
    total_variance_local=np.concatenate((total_variance_local,variance_5.detach().numpy()),axis=0)





    precision_1=1/variance_1
    precision_2=1/variance_2
    precision_3=1/variance_3
    precision_4=1/variance_4
    precision_5=1/variance_5
       
    total_num=total_num+(mean_1.detach().numpy()*precision_1.detach().numpy())
    total_num=total_num+(mean_2.detach().numpy()*precision_2.detach().numpy())
    total_num=total_num+(mean_3.detach().numpy()*precision_3.detach().numpy())
    total_num=total_num+(mean_4.detach().numpy()*precision_4.detach().numpy())
    total_num=total_num+(mean_5.detach().numpy()*precision_5.detach().numpy())
    




    total_precision=total_precision+precision_1.detach().numpy()
    total_precision=total_precision+precision_2.detach().numpy()
    total_precision=total_precision+precision_3.detach().numpy()
    total_precision=total_precision+precision_4.detach().numpy()
    total_precision=total_precision+precision_5.detach().numpy()
    



    

    final_mean=total_num/total_precision
    
    final_precision=1/total_precision
    
    f_=np.concatenate((final_mean,f),axis=1)
    
    estimate=np.concatenate((estimate,f_),axis=0)
    
    
    
    
    
    #f_=np.concatenate((mean.detach().numpy(),f),axis=1) #mean.detach.numpy()
    #estimate=np.concatenate((estimate,f_),axis=0)
    

np.save("bn_vp_5_test_weightmean.npy",estimate)
'''
#colors=['lightgreen','lightblue','red','cyan','orange','purple']
'''
colors=['lightblue','deepskyblue','dodgerblue','cornflowerblue','royalblue','navy']

n,bins,patches=plt.hist(rt60_hist[1:,:],25,stacked=True,density=True,color=colors,label=['125','250','500','1000','2000','4000'])
plt.legend()
plt.savefig("hist_testset_rt60.png")

plt.clf()
n,bins,patches=plt.hist(ab_hist[1:,:],25,stacked=True,density=True,color=colors,label=['125 ab','250 ab','500 ab','1000 ab','2000 ab','4000 ab'])
plt.legend()
plt.savefig("hist_testset_ab.png")



plt.clf()
n,bins,patches=plt.hist(suf_vol[1:,:],20,stacked=True,density=True,color=['lightgreen','lightblue'],label=['Surface','Volume'])
plt.legend()
plt.savefig("hist_testset_surface_vol.png")

'''


