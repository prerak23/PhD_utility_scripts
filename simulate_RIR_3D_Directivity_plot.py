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
#az=100,col=70 D4_1100

'''
dir_obj_mic = DIRPATRir(
    orientation=DirectionVector(azimuth=0,colatitude=0 , degrees=True),
    path="/home/psrivastava/pyroomacoustics/pyroomacoustics/AKG_c480_c414_CUBE.sofa",
    DIRPAT_pattern_enum="AKG_c414K",
    fs=16000,
    #no_points_on_fibo_sphere=0

)
'''

dir_obj_Dmic_1 = DIRPATRir(
    orientation=DirectionVector(azimuth=0,colatitude=0 , degrees=True),
    path="/home/psrivastava/pyroomacoustics/pyroomacoustics/EM32_Directivity.sofa",
    DIRPAT_pattern_enum="EM_32_6",
    fs=16000,
    #no_points_on_fibo_sphere=0

)

dir_obj_Dmic_2 = DIRPATRir(
    orientation=DirectionVector(azimuth=0,colatitude=0 , degrees=True),
    path="/home/psrivastava/pyroomacoustics/pyroomacoustics/EM32_Directivity.sofa",
    DIRPAT_pattern_enum="EM_32_10",
    fs=16000,
    #no_points_on_fibo_sphere=0
)


'''
dir_obj_mic_1 = DIRPATRir(
    orientation=DirectionVector(azimuth=0,colatitude=0 , degrees=True),
    path="/home/psrivastava/pyroomacoustics/pyroomacoustics/AKG_c480_c414_CUBE.sofa",
    directivity_pattern=1,
    fs=16000,
    #no_points_on_fibo_sphere=0

)
'''

'''
dir_obj_mic_2 = DIRPATRir(
    orientation=DirectionVector(azimuth=0,colatitude=0 , degrees=True),
    path="/home/psrivastava/pyroomacoustics/pyroomacoustics/AKG_c480_c414_CUBE.sofa",
    directivity_pattern=1,
    fs=16000,
    #no_points_on_fibo_sphere=0

)
dir_obj_mic_3 = DIRPATRir(
    orientation=DirectionVector(azimuth=0,colatitude=0 , degrees=True),
    path="/home/psrivastava/pyroomacoustics/pyroomacoustics/AKG_c480_c414_CUBE.sofa",
    directivity_pattern=1,
    fs=16000,
    #no_points_on_fibo_sphere=0

)
dir_obj_mic_4 = DIRPATRir(
    orientation=DirectionVector(azimuth=0,colatitude=0 , degrees=True),
    path="/home/psrivastava/pyroomacoustics/pyroomacoustics/AKG_c480_c414_CUBE.sofa",
    directivity_pattern=1,
    fs=16000,
    #no_points_on_fibo_sphere=0

)
dir_obj_mic_5 = DIRPATRir(
    orientation=DirectionVector(azimuth=0,colatitude=0 , degrees=True),
    path="/home/psrivastava/pyroomacoustics/pyroomacoustics/AKG_c480_c414_CUBE.sofa",
    directivity_pattern=1,
    fs=16000,
    #no_points_on_fibo_sphere=0

)
'''


#/home/psrivastava/pyroomacoustics/pyroomacoustics/LSPs_HATS_GuitarCabinets_Akustikmessplatz.sofa

'''
dir_obj_sr = DIRPATRir(
    orientation=DirectionVector(azimuth=0,colatitude=0, degrees=True),
    path="/home/psrivastava/Debug_sofa_file_source_2.sofa",
    directivity_pattern=2,
    fs=16000,
    #no_points_on_fibo_sphere=0
)
'''

"""
dir_obj_sr_1 = DIRPATRir(
    orientation=DirectionVector(azimuth=0,colatitude=0, degrees=True),
    path="/home/psrivastava/pyroomacoustics/pyroomacoustics/LSPs_HATS_GuitarCabinets_Akustikmessplatz.sofa",
    directivity_pattern=0,
    fs=16000,
    #no_points_on_fibo_sphere=0

)
"""


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
    orientation=DirectionVector(azimuth=0,colatitude=0, degrees=True),
    pattern_enum=DirectivityPattern.HYPERCARDIOID,
)
'''

'''
dir_obj_src = CardioidFamily(
    orientation=DirectionVector(azimuth=100,colatitude=70, degrees=True),
    pattern_enum=DirectivityPattern.CARDIOID,
)
'''

dir_obj_Dsrc = DIRPATRir(
    orientation=DirectionVector(azimuth=0, colatitude=0, degrees=True),
    path="/home/psrivastava/pyroomacoustics/pyroomacoustics/LSPs_HATS_GuitarCabinets_Akustikmessplatz.sofa",
    DIRPAT_pattern_enum="HATS_4128C",
    fs=16000,
)



#CARDIOID ,FIGURE_EIGHT,HYPERCARDIOID ,OMNI,SUBCARDIOID
#Hyper-Cardioid : 3
#Figure of eight : 4
#Omni : ?
#Wide Cardioid : 2
#Cardioid : 0


start= timer()
room_dim=[5,5,3]

all_materials = {
    "east": pra.Material(energy_absorption={'coeffs': [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
                                            'center_freqs': [125, 250, 500, 1000, 2000, 4000]},
                         scattering=0.54),
    "west": pra.Material(energy_absorption={'coeffs': [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
                                            'center_freqs': [125, 250, 500, 1000, 2000, 4000]},
                         scattering=0.54),

    "north": pra.Material(energy_absorption={'coeffs': [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
                                             'center_freqs': [125, 250, 500, 1000, 2000, 4000]},
                          scattering=0.54),

    "south": pra.Material(energy_absorption={'coeffs': [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
                                             'center_freqs': [125, 250, 500, 1000, 2000, 4000]},
                          scattering=0.54),

    "ceiling": pra.Material(energy_absorption={'coeffs': [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
                                               'center_freqs': [125, 250, 500, 1000, 2000, 4000]},
                            scattering=0.54),

    "floor": pra.Material(energy_absorption={'coeffs': [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
                                             'center_freqs': [125, 250, 500, 1000, 2000, 4000]},
                          scattering=0.54)}

#Length of the RIR Is 600 ms

room=pra.ShoeBox(room_dim,fs=16000,max_order=20,materials=all_materials,air_absorption=True,ray_tracing=False,min_phase=False)#,pyroom_IWAENC=True)#,min_phase=False)



room.add_source([5/2,5/2,3/2],directivity=dir_obj_Dsrc)#directivity=dir_obj_sr_1) #3.65,1.004,1.38 #0.02,2.004,2.38

#room.add_source([1.52,0.883,1.044])

#mic_locs = np.c_[[5/2+(0.15),5/2+(0.5),3/2],[5/2,5/2+(0.5),3/2],[5/2-(0.15),5/2+(0.5),3/2]]
#voicehome arr mic 1 and mic 2
#mic1=np.array([0.037,0.056,-0.038])
#mic2=np.array([-0.034,0.056,0.038])
#mic1=np.array([0.02432,0.02432,0.024090])
#mic2=np.array([0.02432,-0.02432,-0.024090])


#mic1=bc
#mic2=bc+np.array([0.30,0,0])
#print(mic1,mic2)

#mic1=np.array([0.02432,0.02432,0.024090])
#mic2=np.array([0.02432,-0.02432,-0.024090])

bc=np.array([5/2,3/2,3/2])

mic1=bc
mic2=bc

mic_locs = np.c_[mic1,mic2]#[3.39,2.43,1.31],[3.42,2.48,1.31],[3.47,2.57,1.31],[3.77,3.96,0.91]             #, [3.42, 2.48, 0.91],  # mic 1  # mic 2  #[3.47, 2.57, 1.31], [3.42, 2.48, 0.91]]

room.add_microphone_array(mic_locs,directivity=[dir_obj_Dmic_1,dir_obj_Dmic_2])

#dir_obj_mic.obj_open_sofa_inter.change_orientation(23,43)

#dir_obj_mic_1.change_orientation(np.radians(23),np.radians(43))


dir_obj_Dsrc.obj_open_sofa_inter.change_orientation(230,0) #orientation for the source directive pattern


#room.set_ray_tracing()
room.compute_rir()


end= timer()
plt.clf()


print("Time taken",end-start)



rir_1_0 = room.rir[0][0]

rir_2_0 = room.rir[1][0]

#rir_3_0 = room.rir[2][0]
#rir_3_0 = room.rir[4][1]
#rir_5_0 = room.rir[0][1]

#plt.plot(np.arange(rir_1_0.shape[0]),rir_1_0,label="0,0")
#plt.plot(np.arange(rir_2_0.shape[0]),rir_2_0,label="5,1")
#plt.plot(np.arange(rir_3_0.shape[0]),rir_3_0,label="4,1")
#plt.plot(np.arange(rir_5_0.shape[0]),rir_5_0,label="3,0")
#plt.legend()
#plt.savefig("EM32_RIR.jpg")

#np.save("debug_rir_dir_2.npy",rir_1_0)
np.save("/home/psrivastava/EM32_ref_rir_m1_voicehome2.npy",rir_1_0)
np.save("/home/psrivastava/EM32_ref_rir_m2_voicehome2.npy",rir_2_0)
#np.save("D1_0000_ref_rir_m3_ARR1.npy",rir_2_0)




'''
####################################################
# 3D acoustic scene plotting code                  #
# with directivity pattern for source and receiver #
####################################################
Require position of source and receiver.
Frequency domain filters from interpolated fibo sphere

'''

'''
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('ggplot')


mic_p_x=dir_obj_Dmic_1.obj_open_sofa_inter.rotated_fibo_points[0,:]/12 + np.array([[mic1[0]]])
mic_p_y=dir_obj_Dmic_1.obj_open_sofa_inter.rotated_fibo_points[1,:]/12 + np.array([[mic1[1]]])
mic_p_z=dir_obj_Dmic_1.obj_open_sofa_inter.rotated_fibo_points[2,:]/14 + np.array([[mic1[2]]])


mic_p_x1=dir_obj_Dmic_2.obj_open_sofa_inter.rotated_fibo_points[0,:]/12 + np.array([[mic2[0]]])
mic_p_y1=dir_obj_Dmic_2.obj_open_sofa_inter.rotated_fibo_points[1,:]/12 + np.array([[mic2[1]]])
mic_p_z1=dir_obj_Dmic_2.obj_open_sofa_inter.rotated_fibo_points[2,:]/14 + np.array([[mic2[2]]])


src_p_x=dir_obj_Dsrc.obj_open_sofa_inter.rotated_fibo_points[0,:]/4 +np.array([[5/2]])
src_p_y=dir_obj_Dsrc.obj_open_sofa_inter.rotated_fibo_points[1,:]/4 +np.array([[5/2]])
src_p_z=dir_obj_Dsrc.obj_open_sofa_inter.rotated_fibo_points[2,:]/6  +np.array([[3/2]])







#src_p=dir_obj_sr.obj_open_sofa_inter.rotated_fibo_points+(np.array([[3.65],[1.004],[1.38]]))

#print(np.sqrt(src_p[0,:]**2+src_p[1,:]**2+src_p[2,:]**2))
#print(src_p.shape)

import matplotlib.pyplot as plt

fig=plt.figure(figsize=(8,8))
ax=fig.add_subplot(111,projection='3d')
x=6.4
y=2.9
z=2.8
X, Y, Z = cuboid_data([x/2,y/2,z/2], (x, y, z))
ax.plot_surface(np.array(X), np.array(Y), np.array(Z), color='deepskyblue', rstride=1, cstride=1, alpha=0.05,linewidth=1)


#ax.plot3D([0,x,x,0,0,0,x,x,0,0,0,0,x,x,x,x],
#          [0,0,y,y,0,0,0,y,y,y,0,y,y,y,0,0],
#          [0,0,0,0,0,z,z,z,z,0,z,z,0,z,0,z])


#scamap = plt.cm.ScalarMappable(cmap='summer')
#fcolors_ = scamap.to_rgba(np.abs(fft(dir_obj_sr.obj_open_sofa_inter.sh_coeffs_expanded_target_grid,axis=-1)[:,300]),norm=False)
#fcolors_1 = scamap.to_rgba(np.abs(fft(dir_obj_mic.obj_open_sofa_inter.sh_coeffs_expanded_target_grid,axis=-1)[:,30]),norm=False)
#ax.scatter(mic1[0],mic1[1],mic1[2],c="orange")
#ax.scatter(mic2[0],mic2[1],mic2[2],c="orange")
ax.scatter(src_p_x,src_p_y,src_p_z,c=np.abs(fft(dir_obj_Dsrc.obj_open_sofa_inter.sh_coeffs_expanded_target_grid,axis=-1)[:,30]), cmap='inferno') #sh_coeffs_expanded_target_grid[:,300],fft(dir_obj_sr.obj_open_sofa_inter.sh_coeffs_expanded_target_grid,axis=-1)
ax.scatter(mic_p_x,mic_p_y,mic_p_z,c=np.abs(fft(dir_obj_Dmic_1.obj_open_sofa_inter.sh_coeffs_expanded_target_grid,axis=-1)[:,30]), cmap='inferno')
ax.scatter(mic_p_x1,mic_p_y1,mic_p_z1,c=np.abs(fft(dir_obj_Dmic_2.obj_open_sofa_inter.sh_coeffs_expanded_target_grid,axis=-1)[:,30]), cmap='inferno')

#ax.scatter(src_p_x,src_p_y,src_p_z,c=np.abs(dir_obj_sr.obj_open_sofa_inter.freq_angles_fft[:,30]), cmap='inferno')

#ax.scatter(mic_p_x,mic_p_y,mic_p_z,c=np.abs(dir_obj_mic.obj_open_sofa_inter.freq_angles_fft[:,30]), cmap='inferno')
#ax.scatter(mic_p_x1,mic_p_y1,mic_p_z1,c=np.abs(fft(dir_obj_mic.obj_open_sofa_inter.sh_coeffs_expanded_target_grid,axis=-1)[:,30]), cmap='inferno')



#ax.text(2.31+0.5,1.65+0.5,1.163+0.5,  '%s' % ("(Mic az=0,col=0 , pos = [2.31,1.65,1.163])"), size=8, zorder=1, color='k')
#ax.text(1.52+0.5,0.883+0.5,1.044+0.5,  '%s' % ("(Src az=45,col=0 , pos = [1.52,0.883,1.044])"), size=8, zorder=1, color='k')
ax.set_xlabel('Length')
ax.set_xlim(0, 8)
ax.set_ylabel('Width')
ax.set_ylim(0, 8)
ax.set_zlabel('Height')
ax.set_zlim(0, 3)
plt.legend()
plt.savefig("3d_plot_2.pdf")
#fft(dir_obj_sr.obj_open_sofa_inter.sh_coeffs_expanded_target_grid,axis=-1)

plt.clf()
plt.scatter(np.rad2deg(dir_obj_Dmic_1.obj_open_sofa_inter.phi_fibo),np.rad2deg(dir_obj_Dmic_1.obj_open_sofa_inter.theta_fibo),c=np.abs(fft(dir_obj_Dmic_1.obj_open_sofa_inter.sh_coeffs_expanded_target_grid,axis=-1)[:,2]),cmap="inferno")
plt.colorbar()
plt.savefig("Scatter_plot_em_m6.pdf")
plt.clf()
plt.scatter(np.rad2deg(dir_obj_Dmic_2.obj_open_sofa_inter.phi_fibo),np.rad2deg(dir_obj_Dmic_2.obj_open_sofa_inter.theta_fibo),c=np.abs(fft(dir_obj_Dmic_2.obj_open_sofa_inter.sh_coeffs_expanded_target_grid,axis=-1)[:,2]),cmap="inferno")
plt.colorbar()
plt.savefig("Scatter_plot_em_m10.pdf")
'''

'''
####################################################
# Compare Realism , The code below                 #
# plots dEchorate RIR and pyroom DIRPAT RIR        #
# in time domain and frequency domain              #
####################################################
Requires dirpat pyroom RIR and path to dEchorate RIR
from one of it's rooms.

'''

'''
plt.clf()
plt.style.use('ggplot')
fig,axs=plt.subplots(3,1,figsize=(25,25)) #25,35
#rir_3_0=np.load("debug_rir_dir_3.npy")
rir_2_0=np.load("room011111_2_14.npy").reshape([240000])

print(rir_2_0.shape)
rir_2_0=decimate(rir_2_0,int(round(48000/16000)))[1350:7960]
print(rir_2_0.shape)
AB_=10 * np.log10(np.abs(fft(rir_2_0)[:(rir_2_0.shape[0] // 2)]) ** 2)
BA_=10 * np.log10(np.abs(fft(rir_1_0)[:(rir_1_0.shape[0] // 2)]) ** 2)
#CA_=10 * np.log10(np.abs(fft(rir_3_0)[:(rir_3_0.shape[0] // 2)]) ** 2)

axs[2].plot(fftfreq(rir_2_0.shape[0],d=1/16000)[:(rir_2_0.shape[0]//2)],AB_,label="dEchorate Room 011111")
axs[2].plot(fftfreq(rir_1_0.shape[0],d=1/16000)[:(rir_1_0.shape[0]//2)],BA_,label="Simulated dEchorate room 011111 with pyroom DIRPAT")
#axs[4].plot(fftfreq(rir_3_0.shape[0],d=1/16000)[:(rir_3_0.shape[0]//2)],CA_,label="Vanilla pyroomacoustics simulated dEchorate Room 011111")

axs[1].plot(np.arange(rir_1_0.shape[0]),rir_1_0,label="Simulated dEchorate room 011111 with pyroom DIRPAT ")
axs[1].legend()
axs[0].plot(np.arange(rir_2_0.shape[0]),rir_2_0,label="dEchorate Room 011111",color="steelblue")
axs[0].legend()
#axs[2].plot(np.arange(rir_3_0.shape[0]),rir_3_0,label="Vanilla pyroomacoustics simulated dEchorate Room 011111")

plt.legend()
plt.savefig("Test_2.jpg")

'''



'''
####################################################
# Compare Old pyroom acoustic and dirpat pyroom    #
# plot old RIR and pyroom DIRPAT RIR               #
# in time domain and frequency domain              #
####################################################
Requires path to dirpat pyroom RIR and path to old pyroom generate
RIR.
Simulated in the same frequency , rotation of the source and
receiver should be the same the same goes with the directivity pattern,
acoustic scene should be same . DIrectivity pattern should be imported from SOFA file.

'''

'''
plt.clf()

#To plot the newly created sofa files.
rir_1_0=np.load("debug_rir_dir_1.npy")
rir_2_0=np.load("debug_rir_dir_2.npy") #debug_rir_dir_2.npy,room_011111_c_src.npy

#rir_2_0=decimate(rir_2_0, int(round(48000 / 16000)))
#pad_zero_rir=rir_2_0

pad_zero_rir=np.zeros(rir_1_0.shape[0])
pad_zero_rir[:len(rir_2_0)]=rir_2_0


fig,axs=plt.subplots(2,1,figsize=(25,35))

plt.title("SOFA Files containing frequency independent analytic pattern interpolated on fibo sphere , src rotated az=120 col=31, rec rotated az=46,col=163")


axs[0].plot(np.arange(rir_1_0.shape[0]),pad_zero_rir,c="orange",label="Octave band processing ")
axs[0].legend()
#air_decay = np.exp(-0.5 * air_abs[0] * distance_rir)
#ir_diff[128:N+128]*=air_decay


axs[0].plot(np.arange(rir_1_0.shape[0])-64, rir_1_0, label="DFT domain RIR processing")
axs[0].legend()

#"New Min-phase Method IR With " + txt +" directivity from SOFA files"
AB_=10 * np.log10(np.abs(fft(rir_1_0)[:(rir_1_0.shape[0] // 2)]) ** 2)
BA_=10*np.log10(np.abs(fft(pad_zero_rir)[:(pad_zero_rir.shape[0]//2)])**2)
axs[1].plot(fftfreq(rir_1_0.shape[0],d=1/16000)[:(rir_1_0.shape[0]//2)], AB_,label="DFT domain RIR processing")
axs[1].plot(fftfreq(pad_zero_rir.shape[0],d=1/16000)[:(pad_zero_rir.shape[0]//2)], BA_,label="Octave band processing")
axs[1].legend()

print("RMSE db IN FREQ DOMAIN",np.sqrt(np.mean((AB_-BA_)**2)))
plt.savefig("Test.jpeg")


#rir_1_0 = signal.decimate(rir_1_0,int(round(48000/16000)))
#np.save("debug_rir_4.npy",rir_1_0)

'''

####### Fractional Delay Computation Experiment (Interpolation and Look up table) ############
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
