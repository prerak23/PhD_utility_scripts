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

dir_obj_Dsrc = DIRPATRir(
    orientation=DirectionVector(azimuth=0, colatitude=0, degrees=True),
    path="/home/psrivastava/pyroomacoustics/pyroomacoustics/LSPs_HATS_GuitarCabinets_Akustikmessplatz.sofa",
    DIRPAT_pattern_enum="Genelec_8020",
    fs=16000,
)


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
bc=np.array([5/2,3/2,3/2])
mic1=bc
mic2=bc

mic_locs = np.c_[mic1,mic2]#[3.39,2.43,1.31],[3.42,2.48,1.31],[3.47,2.57,1.31],[3.77,3.96,0.91]             #, [3.42, 2.48, 0.91],  # mic 1  # mic 2  #[3.47, 2.57, 1.31], [3.42, 2.48, 0.91]]

room.add_microphone_array(mic_locs,directivity=[dir_obj_Dmic_1,dir_obj_Dmic_2])

#dir_obj_mic.obj_open_sofa_inter.change_orientation(23,43)

#dir_obj_mic_1.change_orientation(np.radians(23),np.radians(43))


#dir_obj_Dsrc.obj_open_sofa_inter.change_orientation(230,0) #orientation for the source directive pattern


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

plt.plot(np.fft.fftfreq(rir_1_0.shape[0],d=1/16000)[:rir_1_0.shape[0]//2],10*np.log10(np.abs(np.fft.fft(rir_1_0)[:rir_1_0.shape[0]//2])),label="0,0")
#plt.plot(np.arange(rir_2_0.shape[0]),rir_2_0,label="5,1")
#plt.plot(np.arange(rir_3_0.shape[0]),rir_3_0,label="4,1")
#plt.plot(np.arange(rir_5_0.shape[0]),rir_5_0,label="3,0")
#plt.legend()
plt.savefig("EM32_RIR_2.jpg")

#np.save("debug_rir_dir_2.npy",rir_1_0)
#np.save("/home/psrivastava/D7_1111_ref_rir_m1_voicehome2.npy",rir_1_0)
#np.save("/home/psrivastava/D7_1111_ref_rir_m2_voicehome2.npy",rir_2_0)
#np.save("D1_0000_ref_rir_m3_ARR1.npy",rir_2_0)
