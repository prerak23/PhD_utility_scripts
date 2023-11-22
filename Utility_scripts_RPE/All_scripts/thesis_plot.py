import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

rt60_anno=h5py.File("rt60_anno_room_20_2_median.hdf5","r")
other_anno=h5py.File("absorption_surface_calcul_2.hdf5","r")

rt60_matrix=np.zeros((1,6))
abs_matrix=np.zeros((1,6))
sv_matrix=np.zeros((1,2))



for room_no in rt60_anno["room_nos"].keys():
    absorp=other_anno['room_nos'][room_no]['absorption'][()]
    surface=other_anno['room_nos'][room_no]['surface_area'][0]
    volume=other_anno['room_nos'][room_no]['volume'][0]
    rt60=rt60_anno['room_nos'][room_no]['rt60'][()].reshape(6)
    print(room_no)
    rt60_matrix=np.concatenate((rt60_matrix,rt60.reshape(1,6)),axis=0)
    abs_matrix=np.concatenate((abs_matrix,absorp.reshape(1,6)),axis=0)
    sv_matrix=np.concatenate((sv_matrix,[[surface,volume]]),axis=0)


sns.set_theme()
from decimal import Decimal

matrix_list=[rt60_matrix,abs_matrix]#,sv_matrix]
names=["rt60","absorption"]#,"sv"]
labels=[['125 Hz','250 Hz','500 Hz','1000 Hz','2000 Hz','4000 Hz'],['125 Hz','250 Hz','500 Hz','1000 Hz','2000 Hz','4000 Hz']]#,["Surface Area","Volume"]]
fig, axs = plt.subplots(2, 2,figsize=(12,8))
for i in range(2):
    colors=['lightblue','deepskyblue','dodgerblue','cornflowerblue','royalblue','navy']
    #plt.figure(figsize=(2.0,2.0))
    interm_var=matrix_list[i]
    n,bins,patches=axs[0,i].hist(interm_var[1:,:],40,stacked=True,density=False,color=colors[:interm_var.shape[1]],label=labels[i])
    if i == 0:
        axs[0,i].set_xlim(xmin=0, xmax = 4)
        xtick_labels = [r"$89 \mathrm{s}$".replace('89',str(item)) for item in axs[0,i].get_xticks()]
        locs=axs[0,i].get_xticks()
        print(locs)
        axs[0,i].set_xticks(locs,xtick_labels)
        axs[0,i].set_title(r"$RT_{60}$",fontsize=15)
    else:
        axs[0,1].set_title(r"$\overline{\alpha}$",fontsize=15)

    ytick_labels = ["%.1e" % Decimal(item) for item in axs[0,i].get_yticks()]

    for j,k in enumerate(ytick_labels):
        exp=k[-1]
        intr=r"$x \times 10^{z}$".replace("x",k[0])
        intr=intr.replace("z",exp)
        ytick_labels[j]=intr

    ytick_labels[0]="0"
    axs[0,i].set_yticks(axs[0,i].get_yticks(),ytick_labels)
    axs[0,i].set_ylabel("Occurrence",fontsize=15)
    axs[0,i].legend()
    #axs[0,i].savefig("hist_trainset_"+names[i]+".png",bbox_inches='tight')


s=sv_matrix[1:,0]
print(np.max(s))
n,bins,patches=axs[1,0].hist(s,40,stacked=False,density=False,color='deepskyblue',label="Surface Area")
ytick_labels = ["%.1e" % Decimal(item) for item in axs[1,0].get_yticks()]
for j,k in enumerate(ytick_labels):
    exp=k[-1]
    intr=r"$x \times 10^{z}$".replace("x",k[0])
    intr=intr.replace("z",exp)
    ytick_labels[j]=intr

ytick_labels[0]="0"

axs[1,0].set_yticks(axs[1,0].get_yticks(),ytick_labels)
axs[1,0].set_ylabel("Occurrence",fontsize=15)
axs[1,0].set_xlabel(r"$\mathrm{m}^{2}$",fontsize=15)
axs[1,0].set_title(r"$S$")


v=sv_matrix[1:,1]
print(np.max(v))
n,bins,patches=axs[1,1].hist(v,40,stacked=False,density=False,color='dodgerblue',label="Volume")
ytick_labels = ["%.1e" % Decimal(item) for item in axs[1,1].get_yticks()]
for j,k in enumerate(ytick_labels):
    exp=k[-1]
    intr=r"$x \times 10^{z}$".replace("x",k[0])
    intr=intr.replace("z",exp)
    ytick_labels[j]=intr

ytick_labels[0]="0"
axs[1,1].set_yticks(axs[1,1].get_yticks(),ytick_labels)
axs[1,1].set_ylabel("Occurrence",fontsize=15)
axs[1,1].set_xlabel(r"$\mathrm{m}^{3}$",fontsize=15)
axs[1,1].set_title(r"$V$")
fig.tight_layout(pad=3)
plt.savefig("hist_trainset_vol_1.pdf",bbox_inches='tight')
