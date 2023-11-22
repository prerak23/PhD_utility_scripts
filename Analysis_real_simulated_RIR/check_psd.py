import matplotlib 
matplotlib.use('Agg')
import numpy as np
from scipy import signal 
import matplotlib.pyplot as plt
import pyroomacoustics as pra

room_dim=[6,6,2.4]


all_materials={
        "east":pra.Material(energy_absorption={'coeffs':[0.02,0.02,0.03,0.03,0.04,0.05],'center_freqs':[88.38,176.77,353.55,707.10,1414.21,2828.42]},scattering=0.54),
        "west":pra.Material(energy_absorption={'coeffs':[0.02,0.02,0.03,0.03,0.04,0.05],'center_freqs':[88.38,176.77,353.55,707.10,1414.21,2828.42]},scattering=0.54),
 
        "north":pra.Material(energy_absorption={'coeffs':[0.02,0.02,0.03,0.03,0.04,0.05],'center_freqs':[88.38,176.77,353.55,707.10,1414.21,2828.42]},scattering=0.54),
 
        "south":pra.Material(energy_absorption={'coeffs':[0.02,0.02,0.03,0.03,0.04,0.05],'center_freqs':[88.38,176.77,353.55,707.10,1414.21,2828.42]},scattering=0.54),
 
        "ceiling":pra.Material(energy_absorption={'coeffs':[0.02,0.02,0.03,0.03,0.04,0.05],'center_freqs':[88.38,176.77,353.55,707.10,1414.21,2828.42]},scattering=0.54),
        
        "floor":pra.Material(energy_absorption={'coeffs':[0.11,0.14,0.37,0.43,0.27,0.25],'center_freqs':[88.38,176.77,353.55,707.10,1414.21,2828.42]},scattering=0.54)}



room=pra.ShoeBox(room_dim,fs=48000,max_order=10,materials=all_materials,air_absorption=True,ray_tracing=True)

room.set_ray_tracing()
room.add_source([3.65,1.004,1.38])
mic_locs = np.c_[
    [3.47, 2.57, 1.31], [3.42, 2.48, 0.91],  # mic 1  # mic 2
]
room.add_microphone_array(mic_locs)
room.compute_rir()

rir_1_0 = room.rir[0][0]

print(pra.constants.get("frac_delay_length")//2)
room=pra.ShoeBox(room_dim,fs=48000,max_order=10,materials=all_materials,air_absorption=True,ray_tracing=False)

#room.set_ray_tracing()
room.add_source([3.65,1.004,1.38])
mic_locs = np.c_[
    [3.47, 2.57, 1.31], [3.42, 2.48, 0.91],  # mic 1  # mic 2
]
room.add_microphone_array(mic_locs)
room.compute_rir()

rir_1_0_nort = room.rir[0][0]




fig,axs=plt.subplots(4,2,figsize=(11,10)) #30,4

#f,pxx=signal.welch(rir_1_0,16000,nperseg=1024)

'''
axs[2,0].semilogy(f,pxx,color="green",label="pyroom rt")
axs[2,0].set_ylabel("PSD [V**2/Hz]")
axs[2,0].set_xlabel("Frequency Hz")
axs[2,0].legend()
'''
abc=np.load("/home/psrivastava/axis-2/axis-2/roomsim_011111_48khz.npy")[0,:]
print(abc.shape)
#abc=signal.decimate(abc,int(round(48000/16000)))

#f,pxx=signal.welch(abc[0,:18000],16000,nperseg=1024)
#abc_1=abc[0,:]
'''
axs[2,1].semilogy(f,pxx,color="orange",label="roomsim rt")
axs[2,1].set_xlabel("Frequency Hz")
axs[2,1].set_ylabel("PSD [V**2/Hz]")
axs[2,1].legend()
'''
#plt.ylabel("PSD [V**2/Hz]")
#plt.savefig("psd_roomsim_011111_48khz.png")


#abc_1=np.load("011111_room.npy")
#abc=signal.decimate(abc,int(round(48000/16000)))

abc_1=np.load("room_011111_c_src.npy")
print(abc_1.shape)
#f,pxx=signal.welch(abc[1481:18444],16000,nperseg=1024)

'''
axs[3,0].semilogy(f,pxx,color="blue",label="orignal dechorate")
axs[3,0].set_xlabel("Frequency Hz")
axs[3,0].set_ylabel("PSD [V**2/Hz]")
'''


#f,pxx=signal.welch(rir_1_0_nort,16000,nperseg=1024)

'''
axs[3,1].semilogy(f,pxx,color="green",label="pyroom nort")
axs[3,1].set_ylabel("PSD [V**2/Hz]")
axs[3,1].set_xlabel("Frequency Hz")
axs[3,1].legend()
'''

fig,axs=plt.subplots(4,1,figsize=(30,10))
axs[0].plot(np.arange(2400),rir_1_0[39:2439],label="pyroom rt")

axs[0].legend()

axs[1].plot(np.arange(2400),rir_1_0_nort[39:2439],label="pyroom no rt")
axs[1].legend()

axs[2].plot(np.arange(2400),abc[:2400],label="roomsim rt")

axs[2].legend()

axs[3].plot(np.arange(2400),abc_1[4619:7019],label="dEchorate")

print(np.argmax(abc),np.argmax(abc),np.argmax(rir_1_0),np.argmax(rir_1_0_nort))

#axs[3].legend()



fig.tight_layout()

plt.savefig("psd_different_methods.png")




