import matplotlib
matplotlib.use('Agg')
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
decho_file=h5py.File("/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/dEchorate/zenodo.org/record/4626590/files/dechorate.hdf5")

src_id=["1","2","3","4","5","6","7"]
sub_plt_idx=[[(0,0),(0,1)],[(1,0),(1,1)],[(2,0),(2,1)],[(3,0),(3,1)],[(4,0),(4,1)],[(5,0),(5,1)],[(6,0),(6,1)]]


room_id="011111"
mic_id=["1","6","11","16","21","26"]
c=0
tmp_rir=[]
fig=plt.figure(figsize=(25,55))
spec2=gridspec.GridSpec(ncols=1,nrows=42,figure=fig)


'''
for mic in mic_id:

    for src in src_id:

        rir=decho_file[room_id]["rir"][src][mic][:,0]
        #tmp_rir.append(rir)

        a1=int(src)-1+7*c
        a2=0

        b1=int(src)-1+7*c
        b2=1
        print(a1,a2,b1,b2)
        ax_1=fig.add_subplot(spec2[a1,a2])
        #ax_2=fig.add_subplot(spec2[b1,b2])
        #ax_1.plot(np.arange(rir.shape[0]),rir,label=src+" "+mic)
        ax_1.plot(np.arange(3000),rir[3000:6000],label=src+" "+mic)
    c+=1
'''

for mic in mic_id:
    for src in src_id:
        rir=decho_file[room_id]["rir"][src][mic][:,0]
        a1=int(src)-1+7*c
        a2=0
        print(a1,src,mic)
        ax_1=fig.add_subplot(spec2[a1,a2])
        ax_1.plot(np.arange(3000),rir[3000:6000])
        ax_1.set_title(src+" "+"mic "+mic)
    c+=1

plt.tight_layout(pad=2.0)

#file_save=np.array(tmp_rir)
plt.legend()
plt.savefig("check_offset_in_011111_mic1_.png")
#np.save("mic_2_all_src_decorate.npy",file_save)
