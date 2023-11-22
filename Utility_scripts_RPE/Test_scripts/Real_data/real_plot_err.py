import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 

sns.set_theme()


std_vol=106.02265
std_surf=84.2240762
std_rt60=[ 0.7793691,0.7605436,0.6995225, 0.7076664, 0.6420753,0.51794204]
std_abs=[0.100825,0.1172430,0.1002776,0.09108845,0.09378748,0.091663032]




abc_1=np.load("diego_data_mn_wm5_estimate_room_000000.npy")[1:,:]   
abc_2=np.load("diego_data_mn_wm5_estimate_room_000001.npy")[1:,:]   
abc_3=np.load("diego_data_mn_wm5_estimate_room_000010.npy")[1:,:]   
abc_4=np.load("diego_data_mn_wm5_estimate_room_000100.npy")[1:,:]   
abc_5=np.load("diego_data_mn_wm5_estimate_room_001000.npy")[1:,:]   
abc_6=np.load("diego_data_mn_wm5_estimate_room_010000.npy")[1:,:]   
abc_7=np.load("diego_data_mn_wm5_estimate_room_011000.npy")[1:,:]   
abc_8=np.load("diego_data_mn_wm5_estimate_room_011100.npy")[1:,:]   
abc_9=np.load("diego_data_mn_wm5_estimate_room_011110.npy")[1:,:]   












err_1=np.abs(abc_1[:,0]*std_surf-abc_1[:,14])
err_2=np.abs(abc_1[:,1]*std_vol-abc_1[:,15])
err_3=np.abs(abc_1[:,2]*std_rt60[0]-abc_1[:,16])
err_4=np.abs(abc_1[:,3]*std_rt60[1]-abc_1[:,17])
err_5=np.abs(abc_1[:,4]*std_rt60[2]-abc_1[:,18])
err_6=np.abs(abc_1[:,5]*std_rt60[3]-abc_1[:,19])
err_7=np.abs(abc_1[:,6]*std_rt60[4]-abc_1[:,20])
err_8=np.abs(abc_1[:,7]*std_rt60[5]-abc_1[:,21])
err_9=np.abs(abc_1[:,8]*std_abs[0]-abc_1[:,22])
err_10=np.abs(abc_1[:,9]*std_abs[1]-abc_1[:,23])
err_11=np.abs(abc_1[:,10]*std_abs[2]-abc_1[:,24])
err_12=np.abs(abc_1[:,11]*std_abs[3]-abc_1[:,25])
err_13=np.abs(abc_1[:,12]*std_abs[4]-abc_1[:,26])
err_14=np.abs(abc_1[:,13]*std_abs[5]-abc_1[:,27])

r_1=np.abs(abc_2[:,0]*std_surf-abc_2[:,14])
r_2=np.abs(abc_2[:,1]*std_vol-abc_2[:,15])
r_3=np.abs(abc_2[:,2]*std_rt60[0]-abc_2[:,16])
r_4=np.abs(abc_2[:,3]*std_rt60[1]-abc_2[:,17])
r_5=np.abs(abc_2[:,4]*std_rt60[2]-abc_2[:,18])
r_6=np.abs(abc_2[:,5]*std_rt60[3]-abc_2[:,19])
r_7=np.abs(abc_2[:,6]*std_rt60[4]-abc_2[:,20])
r_8=np.abs(abc_2[:,7]*std_rt60[5]-abc_2[:,21])
r_9=np.abs(abc_2[:,8]*std_abs[0]-abc_2[:,22])
r_10=np.abs(abc_2[:,9]*std_abs[1]-abc_2[:,23])
r_11=np.abs(abc_2[:,10]*std_abs[2]-abc_2[:,24])
r_12=np.abs(abc_2[:,11]*std_abs[3]-abc_2[:,25])
r_13=np.abs(abc_2[:,12]*std_abs[4]-abc_2[:,26])
r_14=np.abs(abc_2[:,13]*std_abs[5]-abc_2[:,27])



er_1=np.abs(abc_3[:,0]*std_surf-abc_3[:,14])
er_2=np.abs(abc_3[:,1]*std_vol-abc_3[:,15])
er_3=np.abs(abc_3[:,2]*std_rt60[0]-abc_3[:,16])
er_4=np.abs(abc_3[:,3]*std_rt60[1]-abc_3[:,17])
er_5=np.abs(abc_3[:,4]*std_rt60[2]-abc_3[:,18])
er_6=np.abs(abc_3[:,5]*std_rt60[3]-abc_3[:,19])
er_7=np.abs(abc_3[:,6]*std_rt60[4]-abc_3[:,20])
er_8=np.abs(abc_3[:,7]*std_rt60[5]-abc_3[:,21])
er_9=np.abs(abc_3[:,8]*std_abs[0]-abc_3[:,22])
er_10=np.abs(abc_3[:,9]*std_abs[1]-abc_3[:,23])
er_11=np.abs(abc_3[:,10]*std_abs[2]-abc_3[:,24])
er_12=np.abs(abc_3[:,11]*std_abs[3]-abc_3[:,25])
er_13=np.abs(abc_3[:,12]*std_abs[4]-abc_3[:,26])
er_14=np.abs(abc_3[:,13]*std_abs[5]-abc_3[:,27])



tr_1=np.abs(abc_4[:,0]*std_surf-abc_4[:,14])
tr_2=np.abs(abc_4[:,1]*std_vol-abc_4[:,15])
tr_3=np.abs(abc_4[:,2]*std_rt60[0]-abc_4[:,16])
tr_4=np.abs(abc_4[:,3]*std_rt60[1]-abc_4[:,17])
tr_5=np.abs(abc_4[:,4]*std_rt60[2]-abc_4[:,18])
tr_6=np.abs(abc_4[:,5]*std_rt60[3]-abc_4[:,19])
tr_7=np.abs(abc_4[:,6]*std_rt60[4]-abc_4[:,20])
tr_8=np.abs(abc_4[:,7]*std_rt60[5]-abc_4[:,21])
tr_9=np.abs(abc_4[:,8]*std_abs[0]-abc_4[:,22])
tr_10=np.abs(abc_4[:,9]*std_abs[1]-abc_4[:,23])
tr_11=np.abs(abc_4[:,10]*std_abs[2]-abc_4[:,24])
tr_12=np.abs(abc_4[:,11]*std_abs[3]-abc_4[:,25])
tr_13=np.abs(abc_4[:,12]*std_abs[4]-abc_4[:,26])
tr_14=np.abs(abc_4[:,13]*std_abs[5]-abc_4[:,27])

yr_1=np.abs(abc_5[:,0]*std_surf-abc_5[:,14])
yr_2=np.abs(abc_5[:,1]*std_vol-abc_5[:,15])
yr_3=np.abs(abc_5[:,2]*std_rt60[0]-abc_5[:,16])
yr_4=np.abs(abc_5[:,3]*std_rt60[1]-abc_5[:,17])
yr_5=np.abs(abc_5[:,4]*std_rt60[2]-abc_5[:,18])
yr_6=np.abs(abc_5[:,5]*std_rt60[3]-abc_5[:,19])
yr_7=np.abs(abc_5[:,6]*std_rt60[4]-abc_5[:,20])
yr_8=np.abs(abc_5[:,7]*std_rt60[5]-abc_5[:,21])
yr_9=np.abs(abc_5[:,8]*std_abs[0]-abc_5[:,22])
yr_10=np.abs(abc_5[:,9]*std_abs[1]-abc_5[:,23])
yr_11=np.abs(abc_5[:,10]*std_abs[2]-abc_5[:,24])
yr_12=np.abs(abc_5[:,11]*std_abs[3]-abc_5[:,25])
yr_13=np.abs(abc_5[:,12]*std_abs[4]-abc_5[:,26])
yr_14=np.abs(abc_5[:,13]*std_abs[5]-abc_5[:,27])

ur_1=np.abs(abc_6[:,0]*std_surf-abc_6[:,14])
ur_2=np.abs(abc_6[:,1]*std_vol-abc_6[:,15])
ur_3=np.abs(abc_6[:,2]*std_rt60[0]-abc_6[:,16])
ur_4=np.abs(abc_6[:,3]*std_rt60[1]-abc_6[:,17])
ur_5=np.abs(abc_6[:,4]*std_rt60[2]-abc_6[:,18])
ur_6=np.abs(abc_6[:,5]*std_rt60[3]-abc_6[:,19])
ur_7=np.abs(abc_6[:,6]*std_rt60[4]-abc_6[:,20])
ur_8=np.abs(abc_6[:,7]*std_rt60[5]-abc_6[:,21])
ur_9=np.abs(abc_6[:,8]*std_abs[0]-abc_6[:,22])
ur_10=np.abs(abc_6[:,9]*std_abs[1]-abc_6[:,23])
ur_11=np.abs(abc_6[:,10]*std_abs[2]-abc_6[:,24])
ur_12=np.abs(abc_6[:,11]*std_abs[3]-abc_6[:,25])
ur_13=np.abs(abc_6[:,12]*std_abs[4]-abc_6[:,26])
ur_14=np.abs(abc_6[:,13]*std_abs[5]-abc_6[:,27])

ar_1=np.abs(abc_7[:,0]*std_surf-abc_7[:,14])
ar_2=np.abs(abc_7[:,1]*std_vol-abc_7[:,15])
ar_3=np.abs(abc_7[:,2]*std_rt60[0]-abc_7[:,16])
ar_4=np.abs(abc_7[:,3]*std_rt60[1]-abc_7[:,17])
ar_5=np.abs(abc_7[:,4]*std_rt60[2]-abc_7[:,18])
ar_6=np.abs(abc_7[:,5]*std_rt60[3]-abc_7[:,19])
ar_7=np.abs(abc_7[:,6]*std_rt60[4]-abc_7[:,20])
ar_8=np.abs(abc_7[:,7]*std_rt60[5]-abc_7[:,21])
ar_9=np.abs(abc_7[:,8]*std_abs[0]-abc_7[:,22])
ar_10=np.abs(abc_7[:,9]*std_abs[1]-abc_7[:,23])
ar_11=np.abs(abc_7[:,10]*std_abs[2]-abc_7[:,24])
ar_12=np.abs(abc_7[:,11]*std_abs[3]-abc_7[:,25])
ar_13=np.abs(abc_7[:,12]*std_abs[4]-abc_7[:,26])
ar_14=np.abs(abc_7[:,13]*std_abs[5]-abc_7[:,27])

sr_1=np.abs(abc_8[:,0]*std_surf-abc_8[:,14])
print((abc_8[:,0]*std_surf).mean(),(abc_8[:,0]*std_surf).std())
sr_2=np.abs(abc_8[:,1]*std_vol-abc_8[:,15])
print((abc_8[:,1]*std_vol).mean(),(abc_8[:,1]*std_vol).std())

sr_3=np.abs(abc_8[:,2]*std_rt60[0]-abc_8[:,16])
sr_4=np.abs(abc_8[:,3]*std_rt60[1]-abc_8[:,17])
sr_5=np.abs(abc_8[:,4]*std_rt60[2]-abc_8[:,18])
sr_6=np.abs(abc_8[:,5]*std_rt60[3]-abc_8[:,19])
print((abc_8[:,5]*std_rt60[3]).mean(),(abc_8[:,5]*std_rt60[3]).std())

sr_7=np.abs(abc_8[:,6]*std_rt60[4]-abc_8[:,20])
sr_8=np.abs(abc_8[:,7]*std_rt60[5]-abc_8[:,21])
sr_9=np.abs(abc_8[:,8]*std_abs[0]-abc_8[:,22])
sr_10=np.abs(abc_8[:,9]*std_abs[1]-abc_8[:,23])
sr_11=np.abs(abc_8[:,10]*std_abs[2]-abc_8[:,24])
sr_12=np.abs(abc_8[:,11]*std_abs[3]-abc_8[:,25])
print((abc_8[:,11]*std_abs[3]).mean(),(abc_8[:,11]*std_abs[3]).std())
sr_13=np.abs(abc_8[:,12]*std_abs[4]-abc_8[:,26])
sr_14=np.abs(abc_8[:,13]*std_abs[5]-abc_8[:,27])

xr_1=np.abs(abc_9[:,0]*std_surf-abc_9[:,14])
xr_2=np.abs(abc_9[:,1]*std_vol-abc_9[:,15])
xr_3=np.abs(abc_9[:,2]*std_rt60[0]-abc_9[:,16])
xr_4=np.abs(abc_9[:,3]*std_rt60[1]-abc_9[:,17])
xr_5=np.abs(abc_9[:,4]*std_rt60[2]-abc_9[:,18])
xr_6=np.abs(abc_9[:,5]*std_rt60[3]-abc_9[:,19])
xr_7=np.abs(abc_9[:,6]*std_rt60[4]-abc_9[:,20])
xr_8=np.abs(abc_9[:,7]*std_rt60[5]-abc_9[:,21])
xr_9=np.abs(abc_9[:,8]*std_abs[0]-abc_9[:,22])
xr_10=np.abs(abc_9[:,9]*std_abs[1]-abc_9[:,23])
xr_11=np.abs(abc_9[:,10]*std_abs[2]-abc_9[:,24])
xr_12=np.abs(abc_9[:,11]*std_abs[3]-abc_9[:,25])
xr_13=np.abs(abc_9[:,12]*std_abs[4]-abc_9[:,26])
xr_14=np.abs(abc_9[:,13]*std_abs[5]-abc_9[:,27])



fig,axs=plt.subplots(5,2,figsize=(10,25))
#ep=10e-5
shf=False
bplot1=axs[0,0].boxplot([err_11,r_11,er_11,tr_11,yr_11,ur_11,ar_11,sr_11,xr_11],showmeans=True,vert=True,showfliers=shf,patch_artist=True)
axs[0,0].set_xticks([1,2,3,4,5,6,7,8,9])
axs[0,0].set_xticklabels(['000000','000001','000010','000100','001000','010000','011000','011100','011110'],rotation=45)
axs[0,0].set_ylabel("Abs Err Ab")
axs[0,0].set_title("500hz")

bplot2=axs[0,1].boxplot([err_12,r_12,er_12,tr_12,yr_12,ur_12,ar_12,sr_12,xr_12],showmeans=True,vert=True,showfliers=shf,patch_artist=True)
axs[0,1].set_xticks([1,2,3,4,5,6,7,8,9])
axs[0,1].set_xticklabels(['000000','000001','000010','000100','001000','010000','011000','011100','011110'],rotation=45)
axs[0,1].set_ylabel("Abs Err Ab")
axs[0,1].set_title("1000hz")



bplot3=axs[1,0].boxplot([err_13,r_13,er_13,tr_13,yr_13,ur_13,ar_13,sr_13,xr_13],showmeans=True,vert=True,showfliers=shf,patch_artist=True)
axs[1,0].set_xticks([1,2,3,4,5,6,7,8,9])
axs[1,0].set_xticklabels(['000000','000001','000010','000100','001000','010000','011000','011100','011110'],rotation=45)
axs[1,0].set_ylabel("Abs Err Ab")
axs[1,0].set_title("2000hz")

bplot4=axs[1,1].boxplot([err_14,r_14,er_14,tr_14,yr_14,ur_14,ar_14,sr_14,xr_14],showmeans=True,vert=True,showfliers=shf,patch_artist=True)
axs[1,1].set_xticks([1,2,3,4,5,6,7,8,9])
axs[1,1].set_xticklabels(['000000','000001','000010','000100','001000','010000','011000','011100','011110'],rotation=45)
axs[1,1].set_ylabel("Abs Err Ab")
axs[1,1].set_title("4000hz")

bplot5=axs[2,0].boxplot([err_5,r_5,er_5,tr_5,yr_5,ur_5,ar_5,sr_5,xr_5],showmeans=True,vert=True,showfliers=shf,patch_artist=True)
axs[2,0].set_xticks([1,2,3,4,5,6,7,8,9])
axs[2,0].set_xticklabels(['000000','000001','000010','000100','001000','010000','011000','011100','011110'],rotation=45)
axs[2,0].set_ylabel("Abs Err RT60")
axs[2,0].set_title("500hz")

bplot6=axs[2,1].boxplot([err_6,r_6,er_6,tr_6,yr_6,ur_6,ar_6,sr_6,xr_6],showmeans=True,vert=True,showfliers=shf,patch_artist=True)
axs[2,1].set_xticks([1,2,3,4,5,6,7,8,9])
axs[2,1].set_xticklabels(['000000','000001','000010','000100','001000','010000','011000','011100','011110'],rotation=45)
axs[2,1].set_ylabel("Abs Err RT60")
axs[2,1].set_title("1000hz")

bplot7=axs[3,0].boxplot([err_7,r_7,er_7,tr_7,yr_7,ur_7,ar_7,sr_7,xr_7],showmeans=True,vert=True,showfliers=shf,patch_artist=True)
axs[3,0].set_xticks([1,2,3,4,5,6,7,8,9])
axs[3,0].set_xticklabels(['000000','000001','000010','000100','001000','010000','011000','011100','011110'],rotation=45)
axs[3,0].set_ylabel("Abs Err RT60")
axs[3,0].set_title("2000hz")

bplot8=axs[3,1].boxplot([err_8,r_8,er_8,tr_8,yr_8,ur_8,ar_8,sr_8,xr_8],showmeans=True,vert=True,showfliers=shf,patch_artist=True)
axs[3,1].set_xticks([1,2,3,4,5,6,7,8,9])
axs[3,1].set_xticklabels(['000000','000001','000010','000100','001000','010000','011000','011100','011110'],rotation=45)
axs[3,1].set_ylabel("Abs Err RT60")
axs[3,1].set_title("4000hz")

bplot9=axs[4,0].boxplot([err_1,r_1,er_1,tr_1,yr_1,ur_1,ar_1,sr_1,xr_1],showmeans=True,vert=True,showfliers=shf,patch_artist=True)
axs[4,0].set_xticks([1,2,3,4,5,6,7,8,9])
axs[4,0].set_xticklabels(['000000','000001','000010','000100','001000','010000','011000','011100','011110'],rotation=45)
axs[4,0].set_ylabel("Abs Err M2")
axs[4,0].set_title("Surface")

bplot10=axs[4,1].boxplot([err_2,r_2,er_2,tr_2,yr_2,ur_2,ar_2,sr_2,xr_2],showmeans=True,vert=True,showfliers=shf,patch_artist=True)
axs[4,1].set_xticks([1,2,3,4,5,6,7,8,9])
axs[4,1].set_xticklabels(['000000','000001','000010','000100','001000','010000','011000','011100','011110'],rotation=45)
axs[4,1].set_ylabel("Abs Err M3")
axs[4,1].set_title("Volume")


colors=['pink','lightblue','lightgreen','orange','cyan','green','purple','turquoise','skyblue']


for bplot in (bplot1,bplot2,bplot3,bplot4,bplot5,bplot6,bplot7,bplot8,bplot9,bplot10):
    for patch,color in zip(bplot['boxes'],colors):
        patch.set_facecolor(color)


fig.tight_layout(pad=3.0)
#plt.xticks([1,2,3],('Dummy Bnf','bnf','Dummy M','M'))
#plt.title("Absolute Diff Estimated Mean And Target RT60")
plt.savefig("real_data_mn_2.png")



