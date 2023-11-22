import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#plt.rc("text",usetex=True)

#plt.rc("font",family="serif")

#sns.set_theme()

#plt.rcParams["text.usetex"]=Tr

plt.rc('pdf',fonttype=42)
plt.rc("font",family="serif")


std_vol=106.02265
std_surf=84.2240762
std_rt60=[ 0.7793691,0.7605436,0.6995225, 0.7076664, 0.6420753,0.51794204]
std_abs=[0.100825,0.1172430,0.1002776,0.09108845,0.09378748,0.091663032]


mono_mean_ab=[]
bi_mean_ab=[]
mono_mean_rt=[]
bi_mean_rt=[]
mono_mean_s=[]
bi_mean_s=[]
mono_mean_v=[]
bi_mean_v=[]

mono_err_ab=[]
bi_err_ab=[]
mono_err_rt=[]
bi_err_rt=[]
mono_err_s=[]
bi_err_s=[]
mono_err_v=[]
bi_err_v=[]

for i in ["vp_1_test.npy","mn_vp_2_test_weightmean.npy","mn_vp_3_test_weightmean.npy","mn_vp_4_test_weightmean.npy","mn_vp_5_test_weightmean.npy"]:

    abc=np.load(i)
#abc=np.load("vp_1_test.npy")
#abcd=np.load("")
#abcde=np.load("")
#abcdef=np.load("")
#abcdefg=np.load("")
    err_1=np.abs(abc[1:,0]*std_surf-abc[1:,14])

    err_2=np.abs(abc[1:,1]*std_vol-abc[1:,15])
    mono_mean_s.append(np.array(err_1).mean())
    mono_err_s.append(np.array(err_1).std())
    mono_mean_v.append(np.array(err_2).mean())
    mono_err_v.append(np.array(err_2).std())


    err_3=np.abs(abc[1:,2]*std_rt60[0]-abc[1:,16])
    err_4=np.abs(abc[1:,3]*std_rt60[1]-abc[1:,17])
    err_5=np.abs(abc[1:,4]*std_rt60[2]-abc[1:,18])
    err_6=np.abs(abc[1:,5]*std_rt60[3]-abc[1:,19])
    err_7=np.abs(abc[1:,6]*std_rt60[4]-abc[1:,20])
    err_8=np.abs(abc[1:,7]*std_rt60[5]-abc[1:,21])


    err_9=np.abs(abc[1:,8]*std_abs[0]-abc[1:,22])
    err_10=np.abs(abc[1:,9]*std_abs[1]-abc[1:,23])
    err_11=np.abs(abc[1:,10]*std_abs[2]-abc[1:,24])
    err_12=np.abs(abc[1:,11]*std_abs[3]-abc[1:,25])
    err_13=np.abs(abc[1:,12]*std_abs[4]-abc[1:,26])
    err_14=np.abs(abc[1:,13]*std_abs[5]-abc[1:,27])

    k=(np.array(err_3).mean()+np.array(err_4).mean()+np.array(err_5).mean()+np.array(err_6).mean()+np.array(err_7).mean()+np.array(err_8).mean())/6

    mono_mean_rt.append(k)
    k=(np.array(err_3).std()+np.array(err_4).std()+np.array(err_5).std()+np.array(err_6).std()+np.array(err_7).std()+np.array(err_8).std())/6

    mono_err_rt.append(k)
    k=(np.array(err_9).mean()+np.array(err_10).mean()+np.array(err_11).mean()+np.array(err_12).mean()+np.array(err_13).mean()+np.array(err_14).mean())/6

    mono_mean_ab.append(k)
    k=(np.array(err_9).std()+np.array(err_10).std()+np.array(err_11).std()+np.array(err_12).std()+np.array(err_13).std()+np.array(err_14).std())/6

    mono_err_ab.append(k)

print(mono_mean_s)
print(mono_err_s)
print(mono_mean_v)
print(mono_err_v)
print(mono_mean_rt)
print(mono_err_rt)
print(mono_mean_ab)
print(mono_err_ab)

for i in ["bn_vp_1_test.npy","bn_vp_2_test_weightmean.npy","bn_vp_3_test_weightmean.npy","bn_vp_4_test_weightmean.npy","bn_vp_5_test_weightmean.npy"]:

    abc=np.load(i)
#abc=np.load("vp_1_test.npy")
#abcd=np.load("")
#abcde=np.load("")
#abcdef=np.load("")
#abcdefg=np.load("")
    err_1=np.abs(abc[1:,0]*std_surf-abc[1:,14])

    err_2=np.abs(abc[1:,1]*std_vol-abc[1:,15])
    bi_mean_s.append(np.array(err_1).mean())
    bi_err_s.append(np.array(err_1).std())
    bi_mean_v.append(np.array(err_2).mean())
    bi_err_v.append(np.array(err_2).std())


    err_3=np.abs(abc[1:,2]*std_rt60[0]-abc[1:,16])
    err_4=np.abs(abc[1:,3]*std_rt60[1]-abc[1:,17])
    err_5=np.abs(abc[1:,4]*std_rt60[2]-abc[1:,18])
    err_6=np.abs(abc[1:,5]*std_rt60[3]-abc[1:,19])
    err_7=np.abs(abc[1:,6]*std_rt60[4]-abc[1:,20])
    err_8=np.abs(abc[1:,7]*std_rt60[5]-abc[1:,21])

    err_9=np.abs(abc[1:,8]*std_abs[0]-abc[1:,22])
    err_10=np.abs(abc[1:,9]*std_abs[1]-abc[1:,23])
    err_11=np.abs(abc[1:,10]*std_abs[2]-abc[1:,24])
    err_12=np.abs(abc[1:,11]*std_abs[3]-abc[1:,25])
    err_13=np.abs(abc[1:,12]*std_abs[4]-abc[1:,26])
    err_14=np.abs(abc[1:,13]*std_abs[5]-abc[1:,27])

    k=(np.array(err_3).mean()+np.array(err_4).mean()+np.array(err_5).mean()+np.array(err_6).mean()+np.array(err_7).mean()+np.array(err_8).mean())/6
    bi_mean_rt.append(k)
    k=(np.array(err_3).std()+np.array(err_4).std()+np.array(err_5).std()+np.array(err_6).std()+np.array(err_7).std()+np.array(err_8).std())/6
    bi_err_rt.append(k)
    k=(np.array(err_9).mean()+np.array(err_10).mean()+np.array(err_11).mean()+np.array(err_12).mean()+np.array(err_13).mean()+np.array(err_14).mean())/6
    bi_mean_ab.append(k)
    k=(np.array(err_9).std()+np.array(err_10).std()+np.array(err_11).std()+np.array(err_12).std()+np.array(err_13).std()+np.array(err_14).std())/6
    bi_err_ab.append(k)



mono_err_ab=(np.array(mono_err_ab)*1.96)/44.72
print(mono_err_ab.shape)
bi_err_ab=(np.array(bi_err_ab)*1.96)/44.72
mono_err_rt=(np.array(mono_err_rt)*1.96)/44.72
bi_err_rt=(np.array(bi_err_rt)*1.96)/44.72
mono_err_s=(np.array(mono_err_s)*1.96)/44.72
bi_err_s=(np.array(bi_err_s)*1.96)/44.72
mono_err_v=(np.array(mono_err_v)*1.96)/44.72
bi_err_v=(np.array(bi_err_v)*1.96)/44.72


SMALL_SIZE=8
MEDIUM_SIZE=16
BIGGER_SIZE=16

fig, axs = plt.subplots(nrows=1, ncols=4,figsize=(13,5)) #7.5,3
ls="solid"
plt.rc("font",size=16)
plt.rc("axes",titlesize=16)
plt.rc("axes",labelsize=16)
plt.rc("legend",fontsize=10)
plt.rc("figure",titlesize=16)
plt.rc("xtick",labelsize=16)
#plt.rc("ytick",labelsize=14)




#mono_mean_ab=np.around(np.array(mono_mean_ab),decimals=3,out=None)
#bi_mean_ab=np.around(np.array(bi_mean_ab),decimals=3,out=None)
axs[2].plot(np.arange(5),mono_mean_s,marker="o",color="#fc766aff",label='Single Channel',linestyle=ls)
axs[2].fill_between(np.arange(5),(mono_mean_s-(mono_err_s)),(mono_mean_s+(mono_err_s)),color="#fc766aff",alpha=.15)


axs[2].plot(np.arange(5),bi_mean_s,marker="s",color="#5b84b1ff",label='Multi Channel',linestyle=ls)
axs[2].fill_between(np.arange(5),(bi_mean_s-(bi_err_s)),(bi_mean_s+(bi_err_s)),color="#5b84b1ff",alpha=.15)

axs[2].set_xticks([0,1,2,3,4])
axs[2].set_xticklabels(["1",'2','3','4','5'],size=11)
#axs[2].set_ylim(bottom=0)
axs[2].locator_params(axis='y',nbins=4)
l=axs[2].get_yticks()
print(l[:-1])
axs[2].set_yticks(l[:-1])
axs[2].set_yticklabels(["40","45","50","55"],size=11)
axs[2].set_title("S ($m^2$)")
axs[2].set_xlabel("#pos",size=12)
#axs[0].legend()


axs[3].plot(np.arange(5),mono_mean_v,marker='o',color="#fc766aff",linestyle=ls,label="1chan")
axs[3].fill_between(np.arange(5),(mono_mean_v-(mono_err_v)),(mono_mean_v+(mono_err_v)),color="#fc766aff",alpha=.1)


axs[3].plot(np.arange(5),bi_mean_v,marker='s',color="#5b84b1ff",linestyle=ls,label="2chan")
axs[3].errorbar([0],[79.54],yerr=2.55,marker='^',color="mediumseagreen",label="(Genovese et al., 2019)")
#axs[3].fill_between([0],[79.54-2.55],[79.54+2.55],color="mediumseagreen",alpha=.1)

axs[3].fill_between(np.arange(5),(bi_mean_v-(bi_err_v)),(bi_mean_v+(bi_err_v)),color="#5b84b1ff",alpha=.1)


axs[3].set_xticks([0,1,2,3,4])
axs[3].set_xticklabels(["1",'2','3','4','5'],size=11)
#axs[3].set_ylim(bottom=0)
axs[3].locator_params(axis='y',nbins=5)
l=axs[3].get_yticks()
axs[3].set_yticks([50,60,70,80])
axs[3].set_yticklabels(['50','60','70','80'],size=11)
axs[3].set_title("V ($m^3$)")
axs[3].set_xlabel("#pos",size=12)
axs[3].legend()

axs[1].plot(np.arange(5),mono_mean_rt,marker='o',color="#fc766aff",linestyle=ls)
axs[1].fill_between(np.arange(5),(mono_mean_rt-(mono_err_rt)),(mono_mean_rt+(mono_err_rt)),color="#fc766aff",alpha=.1)


axs[1].plot(np.arange(5),bi_mean_rt,marker='s',color="#5b84b1ff",linestyle=ls)
axs[1].fill_between(np.arange(5),(bi_mean_rt-(bi_err_rt)),(bi_mean_rt+(bi_err_rt)),color="#5b84b1ff",alpha=.1)


axs[1].set_xticks([0,1,2,3,4])
axs[1].set_xticklabels(["1",'2','3','4','5'],size=11)
#axs[1].set_ylim(bottom=0)
axs[1].locator_params(axis='y',nbins=6)
l=axs[1].get_yticks()
print(l)
axs[1].set_yticklabels(['','.16','.18','.20','.22','.24',''],size=11) #l[1:-1]
axs[1].set_title("$RT_{60}$(s)")
axs[1].set_xlabel("#pos",size=12)
#axs[2].legend()

axs[0].plot(np.arange(5),mono_mean_ab,marker='o',color="#fc766aff",linestyle=ls,label="SC")
axs[0].fill_between(np.arange(5),(mono_mean_ab-(mono_err_ab)),(mono_mean_ab+(mono_err_ab)),color="#fc766aff",alpha=.1)


axs[0].plot(np.arange(5),bi_mean_ab,marker='s',color="#5b84b1ff",linestyle=ls,label="SC+IC")
axs[0].fill_between(np.arange(5),(bi_mean_ab-(bi_err_ab)),(bi_mean_ab+(bi_err_ab)),color="#5b84b1ff",alpha=.1)


axs[0].set_xticks([0,1,2,3,4])
#axs[0].set_ylim(bottom=0)
l=axs[1].get_yticks()
print(l)

axs[0].set_xticklabels(["1",'2','3','4','5'],size=11)
axs[0].set_title(r'$\bar{\alpha}$')
axs[0].set_xlabel("#pos",size=12)

#axs[0].set_yticks([0.048,0.50,0.052,0.054,0.056,0.058])
axs[0].set_yticklabels(['','.048','.050','.052','.054','.056','.058'],size=11)
#axs[0].legend()
#plt.xlabel("View Points")
#fig.subplots_adjust(top=0.9, left=0.1, right=0.6, bottom=0.12)  # create some space below the plots by increasing the bottom-value
#axs.flatten()[-2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),ncol=2)
#fig.legend(loc="lower center",bbox_to_anchor=(0.2,-0.1),ncol=2)

fig.tight_layout(pad=0.1)
plt.rc("font",size=SMALL_SIZE)
plt.rc("axes",titlesize=MEDIUM_SIZE)
plt.rc("axes",labelsize=MEDIUM_SIZE)
plt.rc("legend",fontsize=6)
plt.rc("figure",titlesize=BIGGER_SIZE)

#plt.savefig("all_vp_err_plt_10.png")
#plt.savefig("all_vp_err_plt_10.pdf")
plt.savefig("chap_rpe_stds.pdf")
