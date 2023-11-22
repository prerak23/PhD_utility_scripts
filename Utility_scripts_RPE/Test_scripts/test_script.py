import numpy as np
import torch
import h5py
import data_loader_test
import mlh_baseline as net
import torch.nn as nn
import torch.optim as optim 

device=torch.device("cpu")
model_1=net.Model_1().to(device="cpu")
#model_2=net.Model_2().to(device="cuda")
model_3=net.Model_3().to(device="cpu")
model_4=net.Ensemble(model_1,model_3,1).to(device="cpu")
model_4.batch_size=1

chkp=torch.load("/home/psrivastava/baseline/scripts/pre_processing/results_bn_exp/mlh_tas_save_best_sh.pt",map_location=device)
#print(chkp['model_dict_2'])
model_1.load_state_dict(chkp['model_dict_1'])
#model_2.load_state_dict(chkp['model_dict_2'])
model_3.load_state_dict(chkp['model_dict_3'])
model_4.load_state_dict(chkp['model_dict_ens'])
optimizer = optim.Adam(model_4.parameters(), lr=0.0001)
optimizer.load_state_dict(chkp['optimizer_dic'])

model_1.eval()
#model_2.eval()
model_3.eval()
model_4.eval()

abc=np.load("/home/psrivastava/baseline/scripts/pre_processing/test_random_ar.npy")

no_of_vps=2

test_data=data_loader_test.load_data_test(no_of_vps)

no_of_rooms=np.arange(18000,19999)

def cal_features(ch_1,ch_2):
    bz=no_of_vps

    enc_ch1 = torch.stft(ch_1.view(bz, -1), n_fft=1536, hop_length=768, return_complex=True)

    enc_ch2 = torch.stft(ch_2.view(bz, -1), n_fft=1536, hop_length=768, return_complex=True)

    f = torch.view_as_real(enc_ch1)

    f = torch.sqrt(f[:, :, :, 0] ** 2 + f[:, :, :, 1] ** 2)  # Magnitude

    # Ipd ild calculation

    cc = enc_ch1 * torch.conj(enc_ch2)
    ipd = cc / torch.abs(cc)
    ipd_ri = torch.view_as_real(ipd)
    ild = torch.log(torch.abs(enc_ch1) + 10e-5) - torch.log(torch.abs(enc_ch2) + 10e-5)

    x2 = torch.cat((ipd_ri[:, :, :, 0], ipd_ri[:, :, :, 1], ild), axis=1)

    print(f.shape, x2.shape)

    return f ,x2


estimate=np.zeros((1,28))
m=nn.AvgPool1d(32,stride=1)

for r in no_of_rooms:
    
    print('room no',r)
    bnf_mixture_ch1,bnf_mixture_ch2,rt60,ab,surf,vol=test_data.return_data(r)
    
    ch1_feat,ch2_feat=cal_features(bnf_mixture_ch1,bnf_mixture_ch2)
    
    #print(torch.isnan(ch1_feat))
    #print(torch.isnan(ch2_feat))
    #print("vp feat ch1",ch1_feat.shape)
    
    #print("vp feat ch2",ch2_feat.shape)


    total_mean_local=np.zeros((1,14))
    
    total_variance_local=np.zeros((1,14))
    
    total_precision=np.zeros((1,14))
    
    total_num=np.zeros((1,14))

    f=np.concatenate((rt60.reshape(1,6),ab.reshape(1,6),surf.reshape(1,1),vol.reshape(1,1)),axis=1)

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
        '''
        print(x1.shape)
        print(x2.shape)
        x=torch.cat((x1,x2),axis=1)
        x=m(x)

        x=x.reshape(1,-1)
        '''
        mean,variance=model_4(feat_1,feat_2)

        print(mean)
        print(variance)
        total_mean_local=np.concatenate((total_mean_local,mean.detach().numpy()),axis=0)
        
        total_variance_local=np.concatenate((total_variance_local,variance.detach().numpy()),axis=0)

        precision=1/variance
        
        total_num=total_num+(mean.detach().numpy()*precision.detach().numpy())

        total_precision=total_precision+precision.detach().numpy()
    
    final_mean=total_num/total_precision
    #final_precision=1/total_precision
    f_=np.concatenate((final_mean,f),axis=1)
    estimate=np.concatenate((estimate,f_),axis=0)


np.save("vp_2_test.npy",estimate)



