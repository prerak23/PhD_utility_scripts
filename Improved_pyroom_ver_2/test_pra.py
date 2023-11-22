import matplotlib

import pyroomacoustics as pra
import numpy as np
import matplotlib.pyplot as plt


from scipy.io import wavfile
from scipy import signal
from scipy.fft import fftfreq,fft
from pyroomacoustics.directivities import (
    DirectivityPattern,
    DirectionVector,
    CardioidFamily,
    DIRPATRir,
)
from timeit import default_timer as timer

from scipy.signal import decimate


#/home/psrivast/PycharmProjects/axis_2_phd/Debug_sofa_file.sofa
#/home/psrivast/PycharmProjects/axis_2_phd/Debug_sofa_file_source.sofa
#/home/psrivast/Téléchargements/AKG_c480_c414_CUBE.sofa
#/home/psrivast/Téléchargements/LSPs_HATS_GuitarCabinets_Akustikmessplatz.sofa

dir_obj_mic = DIRPATRir(
    orientation=DirectionVector(azimuth=33,colatitude=180 , degrees=True),
    path="/home/psrivast/Téléchargements/AKG_c480_c414_CUBE.sofa",
    directivity_pattern=0,
    fs=16000,
    #no_points_on_fibo_sphere=0

)




dir_obj_sr = DIRPATRir(
    orientation=DirectionVector(azimuth=0,colatitude=90, degrees=True),
    path="/home/psrivast/Téléchargements/LSPs_HATS_GuitarCabinets_Akustikmessplatz.sofa",
    directivity_pattern=4,
    fs=16000,
    #no_points_on_fibo_sphere=0

)




def cuboid_data(center, size):



    #suppose axis direction: x: to left; y: to inside; z: to upper
    # get the (left, outside, bottom) point

    o = [a - b / 2 for a, b in zip(center, size)]
    # get the length, width, and height
    l, w, h = size
    x = [[o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in bottom surface
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in upper surface
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in outside surface
         [o[0], o[0] + l, o[0] + l, o[0], o[0]]]  # x coordinate of points in inside surface
    y = [[o[1], o[1], o[1] + w, o[1] + w, o[1]],  # y coordinate of points in bottom surface
         [o[1], o[1], o[1] + w, o[1] + w, o[1]],  # y coordinate of points in upper surface
         [o[1], o[1], o[1], o[1], o[1]],          # y coordinate of points in outside surface
         [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w]]    # y coordinate of points in inside surface
    z = [[o[2], o[2], o[2], o[2], o[2]],                        # z coordinate of points in bottom surface
         [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],    # z coordinate of points in upper surface
         [o[2], o[2], o[2] + h, o[2] + h, o[2]],                # z coordinate of points in outside surface
         [o[2], o[2], o[2] + h, o[2] + h, o[2]]]                # z coordinate of points in inside surface
    return x,y,z



#az=90,col=10


'''
dir_obj_1 = CardioidFamily(
    orientation=DirectionVector(azimuth=90,colatitude=45, degrees=True),
    pattern_enum=DirectivityPattern.OMNI,
)


dir_obj_src = CardioidFamily(
    orientation=DirectionVector(azimuth=90,colatitude=10, degrees=True),
    pattern_enum=DirectivityPattern.CARDIOID,
)

'''




#CARDIOID ,FIGURE_EIGHT,HYPERCARDIOID ,OMNI,SUBCARDIOID
#Hyper-Cardioid : 3
#Figure of eight : 4
#Omni : ?
#Wide Cardioid : 2
#Cardioid : 0
'''
start= timer()
room_dim=[6,6,2.4]

all_materials = {
    "east": pra.Material(energy_absorption={'coeffs': [0.02, 0.02, 0.03, 0.03, 0.04, 0.05],
                                            'center_freqs': [88.38, 176.77, 353.55, 707.10, 1414.21, 2828.42]},
                         scattering=0.54),
    "west": pra.Material(energy_absorption={'coeffs': [0.02, 0.02, 0.03, 0.03, 0.04, 0.05],
                                            'center_freqs': [88.38, 176.77, 353.55, 707.10, 1414.21, 2828.42]},
                         scattering=0.54),

    "north": pra.Material(energy_absorption={'coeffs': [0.02, 0.02, 0.03, 0.03, 0.04, 0.05],
                                             'center_freqs': [88.38, 176.77, 353.55, 707.10, 1414.21, 2828.42]},
                          scattering=0.54),

    "south": pra.Material(energy_absorption={'coeffs': [0.02, 0.02, 0.03, 0.03, 0.04, 0.05],
                                             'center_freqs': [88.38, 176.77, 353.55, 707.10, 1414.21, 2828.42]},
                          scattering=0.54),

    "ceiling": pra.Material(energy_absorption={'coeffs': [0.02, 0.02, 0.03, 0.03, 0.04, 0.05],
                                               'center_freqs': [88.38, 176.77, 353.55, 707.10, 1414.21, 2828.42]},
                            scattering=0.54),

    "floor": pra.Material(energy_absorption={'coeffs': [0.11, 0.14, 0.37, 0.43, 0.27, 0.25],
                                             'center_freqs': [88.38, 176.77, 353.55, 707.10, 1414.21, 2828.42]},
                          scattering=0.54)}

#Length of the RIR Is 600 ms

room=pra.ShoeBox(room_dim,fs=16000,max_order=20,materials=all_materials,air_absorption=True,ray_tracing=False)#,min_phase=True)#,min_phase=False)


room.add_source([3.65,1.004,1.38] ,directivity=dir_obj_sr) #3.65,1.004,1.38 #0.02,2.004,2.38
mic_locs = np.c_[
    [3.47, 2.57, 1.31], [3.42, 2.48, 0.91],  # mic 1  # mic 2
]
room.add_microphone_array(mic_locs,directivity=dir_obj_mic)
#room.set_ray_tracing()
room.compute_rir()


end= timer()
print("Time taken",end-start)

rir_1_0 = room.rir[1][0]
plt.plot(np.arange(rir_1_0.shape[0]),rir_1_0)
plt.show()


#np.save("debug_rir_dir_3.npy",rir_1_0)

'''

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

mic_p=dir_obj_mic.obj_open_sofa_inter.rotated_fibo_points/2+(np.array([[3.47],[2.57],[1.31]]))
src_p=dir_obj_sr.obj_open_sofa_inter.rotated_fibo_points/2+(np.array([[3.65],[1.004],[1.38]]))

#print(np.sqrt(src_p[0,:]**2+src_p[1,:]**2+src_p[2,:]**2))
#print(src_p.shape)
import matplotlib.pyplot as plt

fig=plt.figure(figsize=(8,8))
ax=fig.add_subplot(111,projection='3d')
x=6
y=6
z=2.4
X, Y, Z = cuboid_data([x/2,y/2,z/2], (x, y, z))
ax.plot_surface(np.array(X), np.array(Y), np.array(Z), color='deepskyblue', rstride=1, cstride=1, alpha=0.05,linewidth=1)


#ax.plot3D([0,x,x,0,0,0,x,x,0,0,0,0,x,x,x,x],
#          [0,0,y,y,0,0,0,y,y,y,0,y,y,y,0,0],
#          [0,0,0,0,0,z,z,z,z,0,z,z,0,z,0,z])

ax.scatter(src_p[0,:],src_p[1,:],src_p[2,:],c=np.abs(fft(dir_obj_sr.obj_open_sofa_inter.sh_coeffs_expanded_target_grid,axis=-1)[:,300])) #sh_coeffs_expanded_target_grid[:,300],fft(dir_obj_sr.obj_open_sofa_inter.sh_coeffs_expanded_target_grid,axis=-1)
ax.scatter(mic_p[0,:],mic_p[1,:],mic_p[2,:],c=np.abs(fft(dir_obj_mic.obj_open_sofa_inter.sh_coeffs_expanded_target_grid,axis=-1)[:,30]))
ax.text(3.47+0.5,2.57+0.5,1.31+0.5,  '%s' % ("(Mic az=0,col=0)"), size=10, zorder=1, color='k')
ax.text(3.65+0.5,1.004+0.5,1.38+0.5,  '%s' % ("(Src az=0,col=0)"), size=10, zorder=1, color='k')
ax.set_xlabel('Length')
ax.set_xlim(0, 6)
ax.set_ylabel('Width')
ax.set_ylim(0, 6)
ax.set_zlabel('Height')
ax.set_zlim(0, 2.4)
plt.legend()
plt.show()
#fft(dir_obj_sr.obj_open_sofa_inter.sh_coeffs_expanded_target_grid,axis=-1)



'''
fig,axs=plt.subplots(3,1,figsize=(25,25)) #25,35
#rir_3_0=np.load("debug_rir_dir_3.npy")
rir_2_0=np.load("room_011111_c_src.npy")
print(rir_2_0.shape)
rir_2_0=decimate(rir_2_0,int(round(48000/16000)))[:6642]
print(rir_2_0.shape)
AB_=10 * np.log10(np.abs(fft(rir_2_0)[:(rir_2_0.shape[0] // 2)]) ** 2)
BA_=10 * np.log10(np.abs(fft(rir_1_0)[:(rir_1_0.shape[0] // 2)]) ** 2)
#CA_=10 * np.log10(np.abs(fft(rir_3_0)[:(rir_3_0.shape[0] // 2)]) ** 2)

axs[2].plot(fftfreq(rir_2_0.shape[0],d=1/16000)[:(rir_2_0.shape[0]//2)],AB_,label="dEchorate Room 011111")
axs[2].plot(fftfreq(rir_1_0.shape[0],d=1/16000)[:(rir_1_0.shape[0]//2)],BA_,label="DIRPAT pyroom simulated dEchorate Room 011111")
#axs[4].plot(fftfreq(rir_3_0.shape[0],d=1/16000)[:(rir_3_0.shape[0]//2)],CA_,label="Vanilla pyroomacoustics simulated dEchorate Room 011111")

axs[0].plot(np.arange(rir_1_0.shape[0]),rir_1_0,label="DIRPAT pyroom simulated dEchorate Room 011111")
axs[0].legend()
axs[1].plot(np.arange(rir_1_0.shape[0]),rir_2_0,label="dEchorate Room 011111")
axs[1].legend()
#axs[2].plot(np.arange(rir_3_0.shape[0]),rir_3_0,label="Vanilla pyroomacoustics simulated dEchorate Room 011111")

plt.legend()
plt.show()
'''

'''
#To plot the newly created sofa files.
rir_1_0=np.load("debug_rir_dir_1.npy")
rir_2_0=np.load("debug_rir_dir_2.npy") #debug_rir_dir_2.npy,room_011111_c_src.npy

#rir_2_0=decimate(rir_2_0, int(round(48000 / 16000)))
pad_zero_rir=np.zeros(rir_1_0.shape[0])
pad_zero_rir[:len(rir_2_0)]=rir_2_0


fig,axs=plt.subplots(2,1,figsize=(25,35))

plt.title("SOFA Files containing frequency independent analytic pattern interpolated on fibo sphere , src rotated az=77 el=133, rec rotated az=73,el=77")


axs[0].plot(np.arange(rir_1_0.shape[0]),pad_zero_rir,c="orange",label="Pyroom acoustic code based on octave band processing ")
axs[0].legend()
#air_decay = np.exp(-0.5 * air_abs[0] * distance_rir)
#ir_diff[128:N+128]*=air_decay


axs[0].plot(np.arange(rir_1_0.shape[0])-64, rir_1_0, label="New method based on DFT scale RIR construction")
axs[0].legend()

#"New Min-phase Method IR With " + txt +" directivity from SOFA files"
AB_=10 * np.log10(np.abs(fft(rir_1_0)[:(rir_1_0.shape[0] // 2)]) ** 2)
BA_=10*np.log10(np.abs(fft(pad_zero_rir)[:(pad_zero_rir.shape[0]//2)])**2)
axs[1].plot(fftfreq(rir_1_0.shape[0],d=1/16000)[:(rir_1_0.shape[0]//2)], AB_,label="New method based on DFT scale RIR construction")
axs[1].plot(fftfreq(pad_zero_rir.shape[0],d=1/16000)[:(pad_zero_rir.shape[0]//2)], BA_,label=" Pyroom acoustic code based on octave band processing")
axs[1].legend()

print("RMSE db IN FREQ DOMAIN",np.sqrt(np.mean((AB_-BA_)**2)))
plt.show()


#rir_1_0 = signal.decimate(rir_1_0,int(round(48000/16000)))
#np.save("debug_rir_4.npy",rir_1_0)

'''



'''
from scipy.fft import fft
fdl=80
lut_gran=20
lut_size = (fdl + 1) * lut_gran + 1
fdl2=(fdl - 1) // 2d 
n = np.linspace(-fdl2-1, fdl2 + 1, lut_size)
print(n)
g=[]
k=np.sinc(n)
tau=8
for i in range(81):
    g.append((k[tau] + 0.66 * (k[tau+1] - k[tau])))
    tau+=20
plt.plot(np.arange(81),np.sinc(g))
plt.show()
'''
'''
tau = 0.3  # Fractional delay [samples].
N = 201  # Filter length.
n = np.arange(N)
#print(n)
#print(n-(N-1)/2)
#print(n-(N-1)/2-tau)

# Compute sinc filter.
h = np.sinc(n - (N - 1) / 2 - tau)

# Multiply sinc filter by window
#h *= np.blackman(N)

# Normalize to get unity gain.
#h /= np.sum(h)
#plt.clf()
#plt.plot(np.arange(201),h)
#plt.show()
from scipy.fft import fftfreq,ifft
import math
s=np.zeros(257,dtype=np.complex_)
for a,f in enumerate(fftfreq(512,d=1/16000)):

        l=-2 * 1j * np.pi * f * tau
        s[a]=np.exp(l)

plt.plot(np.arange(257),np.real(s))
plt.show()
'''

from scipy.fft import fft
from scipy.signal import fftconvolve
'''
impulse_resp=np.zeros(32)
impulse_resp[10]=1
alpha=0.55
si=np.sinc(np.arange(0,32)-0.3)
plt.plot(np.arange(32),np.abs(fft(si)))
plt.show()
wd=np.hanning(32)
ls=[]
plt.clf()
for i in range(32):
        ls.append(alpha*si[i]*wd[i])

plt.plot(np.arange(32),ls)
plt.show()
'''


#dl=np.arange(-40,41,1)
#dl=dl-0.282196044921875
#dl=np.sinc(dl)

#wd=np.hanning(81)
#k=fftconvolve(wd,dl,mode="same")
#plt.plot(np.arange(81),np.abs(fft(wd*dl)))
#plt.show()
