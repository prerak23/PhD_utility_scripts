import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

abc=np.load("vp_1_test.npy")[1:,:]
abcd=np.load("vp_2_test.npy")[1:,:]
abcde=np.load("vp_5_test.npy")[1:,:]
abcdef=np.load("vp5_weightmean_test.npy")[1:,:]
abcdefg=np.load("bn_vp_1_test.npy")[1:,:] #bn_vp_1_test
abcdefgh=np.load("bn_vp_5_test_weightmean.npy")[1:,:]
abcdefghi=np.load("vp_1_6sec.npy")[1:,:]
abcdefghij=np.load("vp_1_as_bn_vp_1.npy")[1:,:]


std_vol=106.02265
std_surf=84.2240762
std_rt60=[ 0.7793691,0.7605436,0.6995225, 0.7076664, 0.6420753,0.51794204]
std_abs=[0.100825,0.1172430,0.1002776,0.09108845,0.09378748,0.091663032]


err_1=np.abs(abc[:,0]*std_surf-abc[:,14])

err_2=np.abs(abc[:,1]*std_vol-abc[:,15])
print(np.array(err_1).mean(),np.array(err_1).std())
print(np.array(err_2).mean(),np.array(err_2).std())


err_3=np.abs(abc[:,2]*std_rt60[0]-abc[:,16])
err_4=np.abs(abc[:,3]*std_rt60[1]-abc[:,17])
err_5=np.abs(abc[:,4]*std_rt60[2]-abc[:,18])
err_6=np.abs(abc[:,5]*std_rt60[3]-abc[:,19])
err_7=np.abs(abc[:,6]*std_rt60[4]-abc[:,20])
err_8=np.abs(abc[:,7]*std_rt60[5]-abc[:,21])
print((err_3.mean()+err_4.mean()+err_5.mean()+err_6.mean()+err_7.mean()+err_8.mean())/6)



err_9=np.abs(abc[:,8]*std_abs[0]-abc[:,22])
err_10=np.abs(abc[:,9]*std_abs[1]-abc[:,23])
err_11=np.abs(abc[:,10]*std_abs[2]-abc[:,24])
err_12=np.abs(abc[:,11]*std_abs[3]-abc[:,25])
err_13=np.abs(abc[:,12]*std_abs[4]-abc[:,26])
err_14=np.abs(abc[:,13]*std_abs[5]-abc[:,27])

print((err_9.mean()+err_10.mean()+err_11.mean()+err_12.mean()+err_13.mean()+err_14.mean())/6)
#print((err_3.std()+err_4.std()+err_5.std()+err_6.std()+err_7.std()+err_8.std())/6)
#print((err_9.std()+err_10.std()+err_11.std()+err_12.std()+err_13.std()+err_14.std())/6)


err2_1=np.abs(abcd[:,0]*std_surf-abcd[:,14])
err2_2=np.abs(abcd[:,1]*std_vol-abcd[:,15])
err2_3=np.abs(abcd[:,2]*std_rt60[0]-abcd[:,16])
err2_4=np.abs(abcd[:,3]*std_rt60[1]-abcd[:,17])
err2_5=np.abs(abcd[:,4]*std_rt60[2]-abcd[:,18])
err2_6=np.abs(abcd[:,5]*std_rt60[3]-abcd[:,19])
err2_7=np.abs(abcd[:,6]*std_rt60[4]-abcd[:,20])
err2_8=np.abs(abcd[:,7]*std_rt60[5]-abcd[:,21])
err2_9=np.abs(abcd[:,8]*std_abs[0]-abcd[:,22])
err2_10=np.abs(abcd[:,9]*std_abs[1]-abcd[:,23])
err2_11=np.abs(abcd[:,10]*std_abs[2]-abcd[:,24])
err2_12=np.abs(abcd[:,11]*std_abs[3]-abcd[:,25])
err2_13=np.abs(abcd[:,12]*std_abs[4]-abcd[:,26])
err2_14=np.abs(abcd[:,13]*std_abs[5]-abcd[:,27])

err3_1=np.abs(abcde[:,0]*std_surf-abcde[:,14])
err3_2=np.abs(abcde[:,1]*std_vol-abcde[:,15])
err3_3=np.abs(abcde[:,2]*std_rt60[0]-abcde[:,16])
err3_4=np.abs(abcde[:,3]*std_rt60[1]-abcde[:,17])
err3_5=np.abs(abcde[:,4]*std_rt60[2]-abcde[:,18])
err3_6=np.abs(abcde[:,5]*std_rt60[3]-abcde[:,19])
err3_7=np.abs(abcde[:,6]*std_rt60[4]-abcde[:,20])
err3_8=np.abs(abcde[:,7]*std_rt60[5]-abcde[:,21])
err3_9=np.abs(abcde[:,8]*std_abs[0]-abcde[:,22])
err3_10=np.abs(abcde[:,9]*std_abs[1]-abcde[:,23])
err3_11=np.abs(abcde[:,10]*std_abs[2]-abcde[:,24])
err3_12=np.abs(abcde[:,11]*std_abs[3]-abcde[:,25])
err3_13=np.abs(abcde[:,12]*std_abs[4]-abcde[:,26])
err3_14=np.abs(abcde[:,13]*std_abs[5]-abcde[:,27])


err4_1=np.abs(abcdef[:,0]*std_surf-abcdef[:,14])
err4_2=np.abs(abcdef[:,1]*std_vol-abcdef[:,15])
err4_3=np.abs(abcdef[:,2]*std_rt60[0]-abcdef[:,16])
err4_4=np.abs(abcdef[:,3]*std_rt60[1]-abcdef[:,17])
err4_5=np.abs(abcdef[:,4]*std_rt60[2]-abcdef[:,18])
err4_6=np.abs(abcdef[:,5]*std_rt60[3]-abcdef[:,19])
err4_7=np.abs(abcdef[:,6]*std_rt60[4]-abcdef[:,20])
err4_8=np.abs(abcdef[:,7]*std_rt60[5]-abcdef[:,21])
err4_9=np.abs(abcdef[:,8]*std_abs[0]-abcdef[:,22])
err4_10=np.abs(abcdef[:,9]*std_abs[1]-abcdef[:,23])
err4_11=np.abs(abcdef[:,10]*std_abs[2]-abcdef[:,24])
err4_12=np.abs(abcdef[:,11]*std_abs[3]-abcdef[:,25])
err4_13=np.abs(abcdef[:,12]*std_abs[4]-abcdef[:,26])
err4_14=np.abs(abcdef[:,13]*std_abs[5]-abcdef[:,27])


err5_1=np.abs(abcdefg[:,0]*std_surf-abcdefg[:,14])
err5_2=np.abs(abcdefg[:,1]*std_vol-abcdefg[:,15])
#print(np.array(err5_1).mean(),np.array(err5_1).std())
#print(np.array(err5_2).mean(),np.array(err5_2).std())


err5_3=np.abs(abcdefg[:,2]*std_rt60[0]-abcdefg[:,16])
err5_4=np.abs(abcdefg[:,3]*std_rt60[1]-abcdefg[:,17])
err5_5=np.abs(abcdefg[:,4]*std_rt60[2]-abcdefg[:,18])
err5_6=np.abs(abcdefg[:,5]*std_rt60[3]-abcdefg[:,19])
err5_7=np.abs(abcdefg[:,6]*std_rt60[4]-abcdefg[:,20])
err5_8=np.abs(abcdefg[:,7]*std_rt60[5]-abcdefg[:,21])

print(err5_3.mean(),err5_4.mean(),err5_5.mean(),err5_6.mean(),err5_7.mean(),err5_8.mean())



err5_9=np.abs(abcdefg[:,8]*std_abs[0]-abcdefg[:,22])
err5_10=np.abs(abcdefg[:,9]*std_abs[1]-abcdefg[:,23])
err5_11=np.abs(abcdefg[:,10]*std_abs[2]-abcdefg[:,24])
err5_12=np.abs(abcdefg[:,11]*std_abs[3]-abcdefg[:,25])
err5_13=np.abs(abcdefg[:,12]*std_abs[4]-abcdefg[:,26])
err5_14=np.abs(abcdefg[:,13]*std_abs[5]-abcdefg[:,27])

print(err5_9.mean(),err5_10.mean(),err5_11.mean(),err5_12.mean(),err5_13.mean(),err5_14.mean())

print((err5_3.mean()+err5_4.mean()+err5_5.mean()+err5_6.mean()+err5_7.mean()+err5_8.mean())/6)
print((err5_9.mean()+err5_10.mean()+err5_11.mean()+err5_12.mean()+err5_13.mean()+err5_14.mean())/6)

#print((err5_3.std()+err5_4.std()+err5_5.std()+err5_6.std()+err5_7.std()+err5_8.std())/6)
#print((err5_9.std()+err5_10.std()+err5_11.std()+err5_12.std()+err5_13.std()+err5_14.std())/6)


print("bn up")



err6_1=np.abs(abcdefgh[:,0]*std_surf-abcdefgh[:,14])
err6_2=np.abs(abcdefgh[:,1]*std_vol-abcdefgh[:,15])




err6_3=np.abs(abcdefgh[:,2]*std_rt60[0]-abcdefgh[:,16])
err6_4=np.abs(abcdefgh[:,3]*std_rt60[1]-abcdefgh[:,17])
err6_5=np.abs(abcdefgh[:,4]*std_rt60[2]-abcdefgh[:,18])
err6_6=np.abs(abcdefgh[:,5]*std_rt60[3]-abcdefgh[:,19])
err6_7=np.abs(abcdefgh[:,6]*std_rt60[4]-abcdefgh[:,20])
err6_8=np.abs(abcdefgh[:,7]*std_rt60[5]-abcdefgh[:,21])

print(err6_3.mean(),err6_4.mean(),err6_5.mean(),err6_6.mean(),err6_7.mean(),err6_8.mean())

err6_9=np.abs(abcdefgh[:,8]*std_abs[0]-abcdefgh[:,22])
err6_10=np.abs(abcdefgh[:,9]*std_abs[1]-abcdefgh[:,23])
err6_11=np.abs(abcdefgh[:,10]*std_abs[2]-abcdefgh[:,24])
err6_12=np.abs(abcdefgh[:,11]*std_abs[3]-abcdefgh[:,25])
err6_13=np.abs(abcdefgh[:,12]*std_abs[4]-abcdefgh[:,26])
err6_14=np.abs(abcdefgh[:,13]*std_abs[5]-abcdefgh[:,27])

print(err6_9.mean(),err6_10.mean(),err6_11.mean(),err6_12.mean(),err6_13.mean(),err6_14.mean())



err7_1=np.abs(abcdefghi[:,0]*std_surf-abcdefghi[:,14])
err7_2=np.abs(abcdefghi[:,1]*std_vol-abcdefghi[:,15])



err7_3=np.abs(abcdefghi[:,2]*std_rt60[0]-abcdefghi[:,16])
err7_4=np.abs(abcdefghi[:,3]*std_rt60[1]-abcdefghi[:,17])
err7_5=np.abs(abcdefghi[:,4]*std_rt60[2]-abcdefghi[:,18])
err7_6=np.abs(abcdefghi[:,5]*std_rt60[3]-abcdefghi[:,19])
err7_7=np.abs(abcdefghi[:,6]*std_rt60[4]-abcdefghi[:,20])
err7_8=np.abs(abcdefghi[:,7]*std_rt60[5]-abcdefghi[:,21])
err7_9=np.abs(abcdefghi[:,8]*std_abs[0]-abcdefghi[:,22])
err7_10=np.abs(abcdefghi[:,9]*std_abs[1]-abcdefghi[:,23])
err7_11=np.abs(abcdefghi[:,10]*std_abs[2]-abcdefghi[:,24])
err7_12=np.abs(abcdefghi[:,11]*std_abs[3]-abcdefghi[:,25])
err7_13=np.abs(abcdefghi[:,12]*std_abs[4]-abcdefghi[:,26])
err7_14=np.abs(abcdefghi[:,13]*std_abs[5]-abcdefghi[:,27])
print("6sec")
#print((err7_3.mean()+err7_4.mean()+err7_5.mean()+err7_6.mean()+err7_7.mean()+err7_8.mean())/6)
#print((err7_9.mean()+err7_10.mean()+err7_11.mean()+err7_12.mean()+err7_13.mean()+err7_14.mean())/6)





err8_1=np.abs(abcdefghij[:,0]*std_surf-abcdefghij[:,14])
err8_2=np.abs(abcdefghij[:,1]*std_vol-abcdefghij[:,15])



err8_3=np.abs(abcdefghij[:,2]*std_rt60[0]-abcdefghij[:,16])
err8_4=np.abs(abcdefghij[:,3]*std_rt60[1]-abcdefghij[:,17])
err8_5=np.abs(abcdefghij[:,4]*std_rt60[2]-abcdefghij[:,18])
err8_6=np.abs(abcdefghij[:,5]*std_rt60[3]-abcdefghij[:,19])
err8_7=np.abs(abcdefghij[:,6]*std_rt60[4]-abcdefghij[:,20])
err8_8=np.abs(abcdefghij[:,7]*std_rt60[5]-abcdefghij[:,21])
err8_9=np.abs(abcdefghij[:,8]*std_abs[0]-abcdefghij[:,22])
err8_10=np.abs(abcdefghij[:,9]*std_abs[1]-abcdefghij[:,23])
err8_11=np.abs(abcdefghij[:,10]*std_abs[2]-abcdefghij[:,24])
err8_12=np.abs(abcdefghij[:,11]*std_abs[3]-abcdefghij[:,25])
err8_13=np.abs(abcdefghij[:,12]*std_abs[4]-abcdefghij[:,26])
err8_14=np.abs(abcdefghij[:,13]*std_abs[5]-abcdefghij[:,27])
print((err8_3.mean()+err8_4.mean()+err8_5.mean()+err8_6.mean()+err8_7.mean()+err8_8.mean())/6)
print((err8_9.mean()+err8_10.mean()+err8_11.mean()+err8_12.mean()+err8_13.mean()+err8_14.mean())/6)


'''
print((np.array(err8_3).mean()+np.array(err8_4).mean()+np.array(err8_5).mean()+np.array(err8_6).mean()+np.array(err8_7).mean()+np.array(err8_8).mean())/6)
print((np.array(err8_3).std()+np.array(err8_4).std()+np.array(err8_5).std()+np.array(err8_6).std()+np.array(err8_7).std()+np.array(err8_8).std())/6)

print((np.array(err8_9).mean()+np.array(err8_10).mean()+np.array(err8_11).mean()+np.array(err8_12).mean()+np.array(err8_13).mean()+np.array(err8_14).mean())/6)
print((np.array(err8_9).std()+np.array(err8_10).std()+np.array(err8_11).std()+np.array(err8_12).std()+np.array(err8_13).std()+np.array(err8_14).std())/6)
'''





std_vol=106.02265
std_surf=84.2240762
std_rt60=[ 0.7793691,0.7605436,0.6995225, 0.7076664, 0.6420753,0.51794204]
std_abs=[0.100825,0.1172430,0.1002776,0.09108845,0.09378748,0.091663032]

fig,axs=plt.subplots(7,2,figsize=(10,25))
f=False
#ep=10e-5
bplot1=axs[0,0].boxplot([err_9,err2_9,err3_9,err4_9,err5_9,err6_9,err7_9,err8_9],showmeans=True,vert=True,showfliers=f,patch_artist=True)
axs[0,0].set_xticks([1,2,3,4,5,6,7,8])
axs[0,0].set_xticklabels(['MN 1 VP','MN 2 VP','MN 5 vp','MN WM 5','BN 1 VP','BN WM 5','6 sec 1 VP','2 CH as BN'],rotation=45)
axs[0,0].set_ylabel("Abs Err Ab")
axs[0,0].set_title("125hz")

#print(bplot1["fliers"][1].get_data()[1])




bplot2=axs[0,1].boxplot([err_10,err2_10,err3_10,err4_10,err5_10,err6_10,err7_10,err8_10],showmeans=True,vert=True,showfliers=f,patch_artist=True)
axs[0,1].set_xticks([1,2,3,4,5,6,7,8])
axs[0,1].set_xticklabels(['MN 1 VP','MN 2 VP','MN 5 vp','MN WM 5','BN 1 VP','BN WM 5','6 sec 1 VP','2 CH as BN'],rotation=45)
axs[0,1].set_ylabel("Abs Err Ab")
axs[0,1].set_title("AB Coeff 250hz")


bplot3=axs[1,0].boxplot([err_11,err2_11,err3_11,err4_11,err5_11,err6_11,err7_11,err8_11],showmeans=True,vert=True,showfliers=f,patch_artist=True)
axs[1,0].set_xticks([1,2,3,4,5,6,7,8])
axs[1,0].set_xticklabels(['MN 1 vp','MN 2 vp','MN 5 vp','MN WM 5','BN 1 VP','BN WM 5','6 sec 1 VP','2 CH as BN'],rotation=45)
axs[1,0].set_ylabel("Abs Err Ab")
axs[1,0].set_title("AB Coeff 500hz")


bplot4=axs[1,1].boxplot([err_12,err2_12,err3_12,err4_12,err5_12,err6_12,err7_12,err8_12],showmeans=True,vert=True,showfliers=f,patch_artist=True)
axs[1,1].set_xticks([1,2,3,4,5,6,7,8])
axs[1,1].set_xticklabels(['MN 1 vp','MN 2 vp','MN 5 vp','MN WM 5','BN 1 VP','BN WM 5','6 sec 1 VP','2 CH as BN'],rotation=45)
axs[1,1].set_ylabel("Abs Err Ab")
axs[1,1].set_title("Ab Coeff 1000hz")


bplot5=axs[2,0].boxplot([err_13,err2_13,err3_13,err4_13,err5_13,err6_13,err7_13,err8_13],showmeans=True,vert=True,showfliers=f,patch_artist=True)
axs[2,0].set_xticks([1,2,3,4,5,6,7,8])
axs[2,0].set_xticklabels(['MN 1 vp','MN 2 vp','MN 5 vp','MN WM 5','BN 1 VP','BN WM 5','6 sec 1 VP','2 CH as BN'],rotation=45)
axs[2,0].set_ylabel("Abs Err Ab")
axs[2,0].set_title("Ab Coeff 2000hz")


bplot6=axs[2,1].boxplot([err_14,err2_14,err3_14,err4_14,err5_14,err6_14,err7_14,err8_14],showmeans=True,vert=True,showfliers=f,patch_artist=True)
axs[2,1].set_xticks([1,2,3,4,5,6,7,8])
axs[2,1].set_xticklabels(['MN 1 vp','MN 2 vp','MN 5 vp','MN WM 5','BN 1 VP','BN WM 5','6 sec 1 VP','2 CH as BN'],rotation=45)
axs[2,1].set_ylabel("Abs Err Ab")
axs[2,1].set_title("Ab Coeff 4000hz")


out_rt=False
bplot7=axs[3,0].boxplot([err_3,err2_3,err3_3,err4_3,err5_3,err6_3,err7_3,err8_3],showmeans=True,vert=True,showfliers=out_rt,patch_artist=True)
axs[3,0].set_xticks([1,2,3,4,5,6,7,8])
axs[3,0].set_xticklabels(['MN 1 src','MN 2 vp','MN 5 vp','MN WM 5','BN 1 VP','BN WM 5','6 sec 1 VP','2 CH as BN'],rotation=45)
axs[3,0].set_ylabel("Abs Err Ab")
axs[3,0].set_title("RT 60 125hz")

#print(bplot1["fliers"][1].get_data()[1])




bplot8=axs[3,1].boxplot([err_4,err2_4,err3_4,err4_4,err5_4,err6_4,err7_4,err8_4],showmeans=True,vert=True,showfliers=out_rt,patch_artist=True)
axs[3,1].set_xticks([1,2,3,4,5,6,7,8])
axs[3,1].set_xticklabels(['MN 1 vp','MN 2 vp','MN 5 vp','MN WM 5','BN 1 VP','BN WM 5','6 sec 1 VP','2 CH as BN'],rotation=45)
axs[3,1].set_ylabel("Abs Err Sec")
axs[3,1].set_title("RT 60 250hz")


bplot9=axs[4,0].boxplot([err_5,err2_5,err3_5,err4_5,err5_5,err6_5,err7_5,err8_5],showmeans=True,vert=True,showfliers=out_rt,patch_artist=True)
axs[4,0].set_xticks([1,2,3,4,5,6,7,8])
axs[4,0].set_xticklabels(['MN 1 vp','MN 2 vp','MN 5 vp','MN WM 5','BN 1 VP','BN WM 5','6 sec 1 VP','2 CH as BN'],rotation=45)
axs[4,0].set_ylabel("Abs Err Sec")
axs[4,0].set_title("RT 60 500hz")
        
bplot10=axs[4,1].boxplot([err_6,err2_6,err3_6,err4_6,err5_6,err6_6,err7_6,err8_6],showmeans=True,vert=True,showfliers=out_rt,patch_artist=True)
axs[4,1].set_xticks([1,2,3,4,5,6,7,8])
axs[4,1].set_xticklabels(['MN 1 vp','MN 2 vp','MN 5 vp','MN WM 5','BN 1 VP','BN WM 5','6 sec 1 VP','2 CH as BN'],rotation=45)
axs[4,1].set_ylabel("Abs Err Sec")
axs[4,1].set_title("RT60 1000hz")


bplot11=axs[5,0].boxplot([err_7,err2_7,err3_7,err4_7,err5_7,err6_7,err7_7,err8_7],showmeans=True,vert=True,showfliers=out_rt,patch_artist=True)
axs[5,0].set_xticks([1,2,3,4,5,6,7,8])
axs[5,0].set_xticklabels(['MN 1 vp','MN 2 vp','MN 5 vp','MN WM 5','BN 1 VP','BN WM 5','6 sec 1 VP','2 CH as BN'],rotation=45)
axs[5,0].set_ylabel("Abs Err Sec")
axs[5,0].set_title("RT 60 2000hz")


bplot12=axs[5,1].boxplot([err_8,err2_8,err3_8,err4_8,err5_8,err6_8,err7_8,err8_8],showmeans=True,vert=True,showfliers=out_rt,patch_artist=True)
axs[5,1].set_xticks([1,2,3,4,5,6,7,8])
axs[5,1].set_xticklabels(['MN 1 vp','MN 2 vp','MN 5 vp','MN WM 5','BN 1 VP','BN WM 5','6 sec 1 VP','2 CH as BN'],rotation=45)
axs[5,1].set_ylabel("Abs Err Sec")
axs[5,1].set_title("RT 60 4000hz")

bplot13=axs[6,0].boxplot([err_1,err2_1,err3_1,err4_1,err5_1,err6_1,err7_1,err8_1],showmeans=True,vert=True,showfliers=f,patch_artist=True)
axs[6,0].set_xticks([1,2,3,4,5,6,7,8])
axs[6,0].set_xticklabels(['MN 1 vp','MN 2 vp','MN 5 vp','MN WM 5','BN 1 VP','BN WM 5','6 sec 1 VP','2 CH as BN'],rotation=45)
axs[6,0].set_ylabel("Abs Err M2")
axs[6,0].set_title("Surface")

bplot14=axs[6,1].boxplot([err_2,err2_2,err3_2,err4_2,err5_2,err6_2,err7_2,err8_2],showmeans=True,vert=True,showfliers=f,patch_artist=True)
axs[6,1].set_xticks([1,2,3,4,5,6,7,8])
axs[6,1].set_xticklabels(['MN 1 vp','MN 2 vp','MN 5 vp','MN WM 5','BN 1 VP','BN WM 5','6 sec 1 VP','2 CH as BN'],rotation=45)
axs[6,1].set_ylabel("Abs Err M3")
axs[6,1].set_title("Volume")



colors=['pink','lightblue','lightgreen','orange','cyan','green','purple','turquoise','skyblue']


for bplot in (bplot1,bplot2,bplot3,bplot4,bplot5,bplot6,bplot7,bplot8,bplot9,bplot10,bplot11,bplot12,bplot13,bplot14):
    for patch,color in zip(bplot['boxes'],colors):
        patch.set_facecolor(color)


fig.tight_layout(pad=3.0)
#plt.xticks([1,2,3],('Dummy Bnf','bnf','Dummy M','M'))
#plt.title("Absolute Diff Estimated Mean And Target RT60")
plt.savefig("test_mono_comparasion.png")
