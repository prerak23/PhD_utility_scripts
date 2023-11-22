import matplotlib
matplotlib.use("Agg")
import pyroomacoustics as pra
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal

room_dim=[6,6,2.4]
#fs,audio=wavfile.read("/home/psrivastava/axis-2/axis-2/clap-single-44110.wav")

#m=pra.make_materials(ceiling="hard_surface",floor="carpet_hairy",east="brickwork",west="brickwork"
#,north="brickwork",south="brickwork")
#print(m)
#ms=pra.Material(energy_absorption={'description':['ceiling','floor','east','west','north','south'],'coeffs':[0.01,0.02,0.02,0.03,0.03,0.04,0.04,0.01,0.02,0.02,0.03,0.03,0.04,0.04,0.01,0.02,0.02,0.03,0.03,0.04,0.04,0.01,0.02,0.02,0.03,0.03,0.04,0.04,0.01,0.02,0.02,0.03,0.03,0.04,0.04,0.01,0.02,0.02,0.03,0.03,0.04,0.04],'center_freqs':[88.38,176.77,353.55,707.10,1414.21,2828.42,5656.85,88.38,176.77,353.55,707.10,1414.21,2828.42,5656.85,88.38,176.77,353.55,707.10,1414.21,2828.42,5656.85,88.38,176.77,353.55,707.10,1414.21,2828.42,5656.85,88.38,176.77,353.55,707.10,1414.21,2828.42,5656.85,88.38,176.77,353.55,707.10,1414.21,2828.42,5656.85]},scattering=0.54)

#m=pra.Material(energy_absorption={'coeffs':[0.01,0.02,0.02,0.03,0.03,0.04],'center_freqs':[88.38,176.77,353.55,707.10,1414.21,2828.42]},scattering=0.54)
#m=pra.make_materials(ceiling={'coeffs':[0.02,0.02,0.03,0.03,0.04,0.05],'center_freqs':[88.38,176.77,353.55,707.10,1414.21,2828.42],'scattering':0.54},floor={'coeffs':[0.11,0.14,0.37,0.43,0.27,0.25],'center_freqs':[88.38,176.77,353.55,707.10,1414.21,2828.42],'scattering':0.54},east={'coeffs':[0.01,0.02,0.02,0.03,0.03,0.04],'center_freqs':[88.38,176.77,353.55,707.10,1414.21,2828.42],'scattering':0.54},west={'coeffs':[0.01,0.02,0.02,0.03,0.03,0.04],'center_freqs':[88.38,176.77,353.55,707.10,1414.21,2828.42],'scattering':0.54},north={'coeffs':[0.01,0.02,0.02,0.03,0.03,0.04],'center_freqs':[88.38,176.77,353.55,707.10,1414.21,2828.42],'scattering':0.54},south={'coeffs':[0.01,0.02,0.02,0.03,0.03,0.04],'center_freqs':[88.38,176.77,353.55,707.10,1414.21,2828.42],'scattering':0.54})

all_materials={
        "east":pra.Material(energy_absorption={'coeffs':[0.02,0.02,0.03,0.03,0.04,0.05],'center_freqs':[88.38,176.77,353.55,707.10,1414.21,2828.42]},scattering=0.54),
        "west":pra.Material(energy_absorption={'coeffs':[0.02,0.02,0.03,0.03,0.04,0.05],'center_freqs':[88.38,176.77,353.55,707.10,1414.21,2828.42]},scattering=0.54),

        "north":pra.Material(energy_absorption={'coeffs':[0.02,0.02,0.03,0.03,0.04,0.05],'center_freqs':[88.38,176.77,353.55,707.10,1414.21,2828.42]},scattering=0.54),

        "south":pra.Material(energy_absorption={'coeffs':[0.02,0.02,0.03,0.03,0.04,0.05],'center_freqs':[88.38,176.77,353.55,707.10,1414.21,2828.42]},scattering=0.54),

        "ceiling":pra.Material(energy_absorption={'coeffs':[0.02,0.02,0.03,0.03,0.04,0.05],'center_freqs':[88.38,176.77,353.55,707.10,1414.21,2828.42]},scattering=0.54),

        "floor":pra.Material(energy_absorption={'coeffs':[0.02,0.02,0.03,0.03,0.04,0.05],'center_freqs':[88.38,176.77,353.55,707.10,1414.21,2828.42]},scattering=0.54)}




#Length of the RIR Is 600 ms

room=pra.ShoeBox(room_dim,fs=48000,max_order=20,materials=all_materials,air_absorption=True,ray_tracing=False)

#room.set_ray_tracing()
room.add_source([3.65,1.004,1.38])
mic_locs = np.c_[
    [3.47, 2.57, 1.31], [3.42, 2.48, 0.91],  # mic 1  # mic 2
]
room.add_microphone_array(mic_locs)
room.compute_rir()

print(room.rir)
rir_1_0 = room.rir[0][0]


#rir_1_0 = signal.decimate(rir_1_0,int(round(48000/16000)))


l=rir_1_0.shape[0]

np.save("pyroom_alpha_check.npy",rir_1_0)


#coeff=np.fft.fft(rir_1_0)

'''
plt.plot(np.fft.fftfreq(l,(1.0/48000.0))[:l//2],2*np.abs(coeff)[:l//2])
plt.xlabel("Frequency Hz")
plt.ylabel("Amplitude")
plt.savefig("011110_pyroomsim_48khz")
'''

'''
f,pxx=signal.welch(rir_1_0,48000,nperseg=1024)
plt.semilogy(f,pxx)
plt.xlabel("Frequency Hz")
plt.ylabel("PSD [V**2/Hz]")
plt.savefig("psd_pyroom_011111_48khz.png")
'''





'''
plt.plot(np.arange(len(rir_1_0)), rir_1_0)

plt.title("The RIR from source 0 to mic 1")
plt.xlabel("Time 1.25 sec ,20000 samples @ fs=16khz ")
plt.savefig("011111_pyra_img.jpeg")
plt.clf()
f,t,Sxx=signal.spectrogram(rir_1_0,16000)
plt.pcolormesh(t,f,np.abs(Sxx))
#plt.yticks(f)
plt.ylabel("Frequency (Hz)")
#plt.xticks(t)
plt.colorbar(mappable=None,use_gridspec=True)
plt.savefig("spec_rt.jpeg")

plt.clf()
f,pxx_den=signal.welch(rir_1_0,16000,nperseg=1024)
plt.semilogy(f,pxx_den)
plt.xlabel('frequench [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.savefig("011111_psd_pyra_img.jpeg")
'''
