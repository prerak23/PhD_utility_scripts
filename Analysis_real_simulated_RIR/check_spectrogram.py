import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal 
import pyroomacoustics as pra

room_dim=[6,6,2.4]


all_materials={
        "east":pra.Material(energy_absorption={'coeffs':[0.02,0.02,0.03,0.03,0.04,0.05],'center_freqs':[88.38,176.77,353.55,707.10,1414.21,2828.42]},scattering=0.54),
        "west":pra.Material(energy_absorption={'coeffs':[0.02,0.02,0.03,0.03,0.04,0.05],'center_freqs':[88.38,176.77,353.55,707.10,1414.21,2828.42]},scattering=0.54),
 
        "north":pra.Material(energy_absorption={'coeffs':[0.02,0.02,0.03,0.03,0.04,0.05],'center_freqs':[88.38,176.77,353.55,707.10,1414.21,2828.42]},scattering=0.54),
 
        "south":pra.Material(energy_absorption={'coeffs':[0.02,0.02,0.03,0.03,0.04,0.05],'center_freqs':[88.38,176.77,353.55,707.10,1414.21,2828.42]},scattering=0.54),
 
        "ceiling":pra.Material(energy_absorption={'coeffs':[0.02,0.02,0.03,0.03,0.04,0.05],'center_freqs':[88.38,176.77,353.55,707.10,1414.21,2828.42]},scattering=0.54),
        
        "floor":pra.Material(energy_absorption={'coeffs':[0.11,0.14,0.37,0.43,0.27,0.25],'center_freqs':[88.38,176.77,353.55,707.10,1414.21,2828.42]},scattering=0.54)}



room=pra.ShoeBox(room_dim,fs=48000,max_order=20,materials=all_materials,air_absorption=True,ray_tracing=True)

room.set_ray_tracing()
room.add_source([3.65,1.004,1.38])
mic_locs = np.c_[
    [3.47, 2.57, 1.31], [3.42, 2.48, 0.91],  # mic 1  # mic 2
]
room.add_microphone_array(mic_locs)
room.compute_rir()

rir_1_0 = room.rir[0][0]




room=pra.ShoeBox(room_dim,fs=48000,max_order=20,materials=all_materials,air_absorption=True,ray_tracing=False)

#room.set_ray_tracing()
room.add_source([3.65,1.004,1.38])
mic_locs = np.c_[
    [3.47, 2.57, 1.31], [3.42, 2.48, 0.91],  # mic 1  # mic 2
]
room.add_microphone_array(mic_locs)
room.compute_rir()

rir_1_0_nort = room.rir[0][0]

global_delay=pra.constants.get("frac_delay_length")//2
print(global_delay)


fig,axs=plt.subplots(2,2,figsize=(12,9)) #figsize=(30,4)

abc=np.load("room_011111_c_src.npy")
#abc=signal.decimate(abc,int(round(48000/16000)))

#abc=np.load("room_011111_c_src.npy")

abc_1=np.load("/home/psrivastava/axis-2/axis-2/roomsim_011111_48khz.npy")[0,:]
#abc_1=signal.decimate(abc_1,int(round(48000/16000)))


#print(abc_1.shape)
#print(abc.shape)
#down_sample=signal.decimate(abc,int(round(48000/16000)))

f,t,Sxx=signal.spectrogram(abc[4619:7019],48000,nperseg=48,noverlap=48//1.3) #4520,6920
f_1,t_1,Sxx_1=signal.spectrogram(abc_1[:2400],48000,nperseg=48,noverlap=48//1.3)
f_2,t_2,Sxx_2=signal.spectrogram(rir_1_0[39:2439],48000,nperseg=48,noverlap=48//1.3)
f_3,t_3,Sxx_3=signal.spectrogram(rir_1_0_nort[29:2439],48000,nperseg=48,noverlap=48//1.3)
print(f)
print(f_1)
print(f_2)
print(f_3)

mf=axs[0,0].pcolormesh(t,f,10*np.log10(np.abs(Sxx)),shading='gouraud')
axs[0,0].set_xlabel("Time [sec]")
axs[0,0].set_ylabel("Frequency [Hz]")
axs[0,0].set_title("dEchorate")
fig.colorbar(mf,use_gridspec=True,ax=axs[0,0])
#plt.savefig("Spectrogram_50ms_dEchorate_k=16ms.png")
#plt.clf()


g=axs[0,1].pcolormesh(t_1,f_1,10*np.log10(np.abs(Sxx_1)),shading='gouraud')
axs[0,1].set_xlabel("Time [sec]")
axs[0,1].set_ylabel("Frequency [Hz]")
axs[0,1].set_title("Roomsim")
fig.colorbar(g,use_gridspec=True,ax=axs[0,1])
#plt.savefig("Spectrogram_50ms_roomsim.png")
#plt.clf()



k=axs[1,0].pcolormesh(t_2,f_2,10*np.log10(np.abs(Sxx_2)),shading='gouraud')
axs[1,0].set_xlabel("Time [sec]")
axs[1,0].set_ylabel("Frequency [Hz]")
axs[1,0].set_title("Pyroom RT")
fig.colorbar(k,use_gridspec=True,ax=axs[1,0])
#plt.savefig("Spectrogram_50ms_pyroom_rt.png")

#plt.clf()

l=axs[1,1].pcolormesh(t_3,f_3,10*np.log10(np.abs(Sxx_3)),shading='gouraud')
axs[1,1].set_xlabel("Time [sec]")
axs[1,1].set_ylabel("Frequency [Hz]")
axs[1,1].set_title("Pyroom NORT")
fig.colorbar(l,use_gridspec=True,ax=axs[1,1])

fig.tight_layout(pad=2.0)

plt.savefig("Spectrogram_32ms_window.png")






#Plot Spectrogram 


plt.clf()
Sxx=10*np.log10(np.abs(Sxx))
Sxx_1=10*np.log10(np.abs(Sxx_1))
Sxx_2=10*np.log10(np.abs(Sxx_2))
Sxx_3=10*np.log10(np.abs(Sxx_3))





fig,axs=plt.subplots(2,2,figsize=(12,9)) #30,4
axs[0,0].hist(Sxx[1,:],density=True,alpha=0.5,label="500Hz")
axs[0,0].hist(Sxx[2,:],density=True,alpha=0.5,label="1000Hz")
axs[0,0].hist(Sxx[3,:],density=True,alpha=0.5,label="1500Hz")
axs[0,0].hist(Sxx[4,:],density=True,alpha=0.5,label="2000Hz")
axs[0,0].hist(Sxx[5,:],density=True,alpha=0.5,label="2500Hz")
axs[0,0].hist(Sxx[6,:],density=True,alpha=0.5,label="3000Hz")
axs[0,0].hist(Sxx[7,:],density=True,alpha=0.5,label="3500Hz")
axs[0,0].hist(Sxx[8,:],density=True,alpha=0.5,label="4000Hz")
axs[0,0].hist(Sxx[9,:],density=True,alpha=0.5,label="4500Hz")
axs[0,0].hist(Sxx[10,:],density=True,alpha=0.5,label="5000Hz")
axs[0,0].hist(Sxx[11,:],density=True,alpha=0.5,label="5500Hz")
axs[0,0].hist(Sxx[12,:],density=True,alpha=0.5,label="6000Hz")
axs[0,0].hist(Sxx[13,:],density=True,alpha=0.5,label="6500Hz")
axs[0,0].hist(Sxx[14,:],density=True,alpha=0.5,label="7000Hz")
axs[0,0].hist(Sxx[15,:],density=True,alpha=0.5,label="7500Hz")
axs[0,0].hist(Sxx[16,:],density=True,alpha=0.5,label="8000Hz")
















axs[0,0].set_xlabel("Magnitude")
axs[0,0].set_ylabel("Counts")
axs[0,0].set_title("dEchorate")

axs[0,1].hist(Sxx_1[1,:],density=True,alpha=0.5,label="500Hz")
axs[0,1].hist(Sxx_1[2,:],density=True,alpha=0.5,label="1000Hz")
axs[0,1].hist(Sxx_1[3,:],density=True,alpha=0.5,label="1500Hz")
axs[0,1].hist(Sxx_1[4,:],density=True,alpha=0.5,label="2000Hz")
axs[0,1].hist(Sxx_1[5,:],density=True,alpha=0.5,label="2500Hz")
axs[0,1].hist(Sxx_1[6,:],density=True,alpha=0.5,label="3000Hz")
axs[0,1].hist(Sxx_1[7,:],density=True,alpha=0.5,label="3500Hz")
axs[0,1].hist(Sxx_1[8,:],density=True,alpha=0.5,label="4000Hz")
axs[0,1].hist(Sxx_1[9,:],density=True,alpha=0.5,label="4500Hz")
axs[0,1].hist(Sxx_1[10,:],density=True,alpha=0.5,label="5000Hz")
axs[0,1].hist(Sxx_1[11,:],density=True,alpha=0.5,label="5500Hz")
axs[0,1].hist(Sxx_1[12,:],density=True,alpha=0.5,label="6000Hz")
axs[0,1].hist(Sxx_1[13,:],density=True,alpha=0.5,label="6500Hz")
axs[0,1].hist(Sxx_1[14,:],density=True,alpha=0.5,label="7000Hz")
axs[0,1].hist(Sxx_1[15,:],density=True,alpha=0.5,label="7500Hz")
axs[0,1].hist(Sxx_1[16,:],density=True,alpha=0.5,label="8000Hz")
















axs[0,1].set_xlabel("Magnitude")
axs[0,1].set_ylabel("Counts")
axs[0,1].set_title("Roomsim")




axs[1,0].hist(Sxx_2[1,:],density=True,alpha=0.5,label="500Hz")
axs[1,0].hist(Sxx_2[2,:],density=True,alpha=0.5,label="1000Hz")
axs[1,0].hist(Sxx_2[3,:],density=True,alpha=0.5,label="1500Hz")
axs[1,0].hist(Sxx_2[4,:],density=True,alpha=0.5,label="2000Hz")
axs[1,0].hist(Sxx_2[5,:],density=True,alpha=0.5,label="2500Hz")
axs[1,0].hist(Sxx_2[6,:],density=True,alpha=0.5,label="3000Hz")
axs[1,0].hist(Sxx_2[7,:],density=True,alpha=0.5,label="3500Hz")
axs[1,0].hist(Sxx_2[8,:],density=True,alpha=0.5,label="4000Hz")
axs[1,0].hist(Sxx_2[9,:],density=True,alpha=0.5,label="4500Hz")
axs[1,0].hist(Sxx_2[10,:],density=True,alpha=0.5,label="5000Hz")
axs[1,0].hist(Sxx_2[11,:],density=True,alpha=0.5,label="5500Hz")
axs[1,0].hist(Sxx_2[12,:],density=True,alpha=0.5,label="6000Hz")
axs[1,0].hist(Sxx_2[13,:],density=True,alpha=0.5,label="6500Hz")
axs[1,0].hist(Sxx_2[14,:],density=True,alpha=0.5,label="7000Hz")
axs[1,0].hist(Sxx_2[15,:],density=True,alpha=0.5,label="7500Hz")
axs[1,0].hist(Sxx_2[16,:],density=True,alpha=0.5,label="8000Hz")
















axs[1,0].set_xlabel("Magnitude")
axs[1,0].set_ylabel("Counts")
axs[1,0].set_title("Pyroom RT")





axs[1,1].hist(Sxx_3[1,:],density=True,alpha=0.5,label="500Hz")
axs[1,1].hist(Sxx_3[2,:],density=True,alpha=0.5,label="1000Hz")
axs[1,1].hist(Sxx_3[3,:],density=True,alpha=0.5,label="1500Hz")
axs[1,1].hist(Sxx_3[4,:],density=True,alpha=0.5,label="2000Hz")
axs[1,1].hist(Sxx_3[5,:],density=True,alpha=0.5,label="2500Hz")
axs[1,1].hist(Sxx_3[6,:],density=True,alpha=0.5,label="3000Hz")
axs[1,1].hist(Sxx_3[7,:],density=True,alpha=0.5,label="3500Hz")
axs[1,1].hist(Sxx_3[8,:],density=True,alpha=0.5,label="4000Hz")
axs[1,1].hist(Sxx_3[9,:],density=True,alpha=0.5,label="4500Hz")
axs[1,1].hist(Sxx_3[10,:],density=True,alpha=0.5,label="5000Hz")
axs[1,1].hist(Sxx_3[11,:],density=True,alpha=0.5,label="5500Hz")
axs[1,1].hist(Sxx_3[12,:],density=True,alpha=0.5,label="6000Hz")
axs[1,1].hist(Sxx_3[13,:],density=True,alpha=0.5,label="6500Hz")
axs[1,1].hist(Sxx_3[14,:],density=True,alpha=0.5,label="7000Hz")
axs[1,1].hist(Sxx_3[15,:],density=True,alpha=0.5,label="7500Hz")
axs[1,1].hist(Sxx_3[16,:],density=True,alpha=0.5,label="8000Hz")
















axs[1,1].set_xlabel("Magnitude")
axs[1,1].set_ylabel("Counts")
axs[1,1].set_title("Pyroom NORT")

fig.tight_layout(pad=2.0)

plt.legend(loc="lower right",bbox_to_anchor=(1.2,0.9))

plt.savefig("Hist_spec.png",bbox_inches='tight')



#plt.plot(np.fft.fftfreq(24000,(1.0/48000.0))[:12000],2*np.abs(coeff_1[:12000]))


'''
plt.xlabel("Frquency Hz")
plt.ylabel("Amplitude")
plt.savefig("24000_point_fft_011111_dEchorate.png")

plt.clf()

plt.plot(np.fft.fftfreq(24000,(1.0/48000.0))[:12000],2*np.abs(coeff_2[:12000]))
plt.ylim(top=8,bottom=0)
plt.xlabel("Frquency Hz")
plt.ylabel("Amplitude")
plt.savefig("24000_point_fft_011111_roomsim.png")

'''



