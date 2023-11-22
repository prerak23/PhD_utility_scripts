import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pyroomacoustics as pra
import h5py
import yaml
import sys 
sys.path.insert(0,"/home/psrivastava/baseline/sofamyroom_2/")
from test_axis_2 import cal_roomsim

def cal_grp_delay(t, f, Sxx):
    grp_delay = []

    for _ in range((t.shape[0])):
        grp_val = []

        k = (-1 / 360) * ((Sxx[1][_] - Sxx[0][_]) / (f[1] - f[0]))
        grp_val.append(k)

        for ft in range(f.shape[0] - 1)[1:]:
            k = (-1 / 720) * (((Sxx[ft][_] - Sxx[ft - 1][_]) / (f[ft] - f[ft - 1])) + (
                        (Sxx[ft + 1][_] - Sxx[ft][_]) / (f[ft + 1] - f[ft])))
            grp_val.append(k)

        sh = f.shape[0] - 1

        k = (-1 / 360) * ((Sxx[sh][_] - Sxx[sh - 1][_]) / (f[sh] - f[sh - 1]))
        grp_val.append(k)
        grp_delay.append(grp_val)

    g_delay = np.array(grp_delay).reshape(f.shape[0], -1)
    return g_delay



def rir_pyroom(sampling_freq,ref_order,src,mic_1,mic_2,mic_3,mic_4,mic_5,mic_6,mic_7,mic_8,mic_9,mic_10,mic_11,mic_12):

    room_dim = [6, 6, 2.4]

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

    room = pra.ShoeBox(room_dim, fs=sampling_freq, max_order=ref_order, materials=all_materials, air_absorption=True, ray_tracing=True)

    room.set_ray_tracing()
    room.add_source(src)
    mic_locs = np.c_[
        mic_1, mic_2, mic_3, mic_4, mic_5,mic_6,mic_7,mic_8,mic_9,mic_10,mic_11,mic_12  # mic 1  # mic 2
    ]
    room.add_microphone_array(mic_locs)
    room.compute_rir()

    rir_1_0 = room.rir[5][0]

    room = pra.ShoeBox(room_dim, fs=sampling_freq, max_order=ref_order, materials=all_materials, air_absorption=True, ray_tracing=False)

    # room.set_ray_tracing()
    room.add_source(src)
    mic_locs = np.c_[
        mic_1, mic_2, mic_3, mic_4, mic_5, mic_6, mic_7, mic_8, mic_9, mic_10, mic_11, mic_12   # mic 1  # mic 2
    ]
    room.add_microphone_array(mic_locs)
    room.compute_rir()
    rir_1_0_nort=room.rir[5][0]
    
    '''
    with open("/home/psrivastava/baseline/sofamyroom_2/conf_source_1.yml",'w') as file_3:
        dicto=yaml.load(file_3,Loader=yaml.FullLoader)
        print(dicto)
        dicto["room_1"]["source_pos"][0]=src
        docs=yaml.dump(dicto,"/home/psrivastava/baseline/sofamyroom_2/conf_source_1.yml")
    '''
    roomsim_np_arr=cal_roomsim(src)

    print(roomsim_np_arr.shape)

    return rir_1_0,rir_1_0_nort,roomsim_np_arr[0,:]


roomsim_rir=np.load("/home/psrivastava/axis-2/axis-2/roomsim_011111_48khz.npy")[0,:]

dEchorate_dataset=h5py.File("/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/dEchorate/zenodo.org/record/4626590/files/dechorate.hdf5")
room_id="011111"

src_id=['1','2','3','4','5','6']
directional_src=[[1.991,4.498,1.424],[1.523,0.883,1.044],[4.287,2.069,1.074],[4.619,3.669,1.504],[1.523,0.723,1.184],[4.447,2.069,1.214]]
array_2_mics=[[[0.8577,3.909,1.039],[0.895,3.924,1.039]],[[0.816,2.244,0.98],[0.8032,2.2065,0.98]],[[2.182,1.726,1.163],[2.218,1.707,1.163]],[[3.342,2.357,1.31],[3.363,2.391,1.31]],[[3.77,3.96,0.91],[3.74,3.93,0.91]],[[3.129,3.372,1.46],[3.097,3.396,1.46]]]
fs_compute=48000

for src_no in src_id:

    rir_rt,rir_no_rt,rir_roomsim=rir_pyroom(fs_compute,20,directional_src[int(src_no)-1],array_2_mics[0][0],array_2_mics[0][1],array_2_mics[1][0],array_2_mics[1][1],array_2_mics[2][0],array_2_mics[2][1],array_2_mics[3][0],array_2_mics[3][1],array_2_mics[4][0],array_2_mics[4][1],array_2_mics[5][0],array_2_mics[5][1])
    sample_50ms=int((fs_compute/1000)*50)
    window_size=2 #in ms
    window_samples=int((fs_compute/1000)*window_size)
    rir_dechorate=dEchorate_dataset[room_id]["rir"][src_no]["11"][:,0]
   
    max_roomsim=np.max(rir_roomsim)
    max_decho=np.max(rir_dechorate)
    diff=max_roomsim-max_decho
    '''
    if diff > 0:
        start_decho=int(max_roomsim-diff)
    '''
    idx_dechorate=int(sample_50ms+4619)
    print(idx_dechorate)
    idx_pyroom=int(sample_50ms+39)
    print(rir_dechorate[4619:idx_dechorate])
    f, t, Sxx = signal.spectrogram(rir_dechorate[4619:idx_dechorate], fs_compute, nperseg=window_samples, noverlap=window_samples // 1.3)  # 4520,6920
    f_1, t_1, Sxx_1 = signal.spectrogram(rir_roomsim[:sample_50ms], fs_compute, nperseg=window_samples, noverlap= window_samples // 1.3)
    f_2, t_2, Sxx_2 = signal.spectrogram(rir_rt[39:idx_pyroom], fs_compute, nperseg=window_samples, noverlap= window_samples // 1.3)
    f_3, t_3, Sxx_3 = signal.spectrogram(rir_no_rt[39:idx_pyroom], fs_compute, nperseg=window_samples, noverlap= window_samples // 1.3)

    #Calculate Phase

    f_a, t_a, Sxx_a = signal.spectrogram(rir_dechorate[4619:idx_dechorate], fs_compute, mode="angle",nperseg=window_samples,
                                   noverlap=window_samples // 1.3)  # 4520,6920
    f_1_a, t_1_a, Sxx_1_a = signal.spectrogram(rir_roomsim[:sample_50ms], fs_compute, mode="angle",nperseg=window_samples,
                                         noverlap=window_samples // 1.3)
    f_2_a, t_2_a, Sxx_2_a = signal.spectrogram(rir_rt[39:idx_pyroom], fs_compute, mode="angle",nperseg=window_samples,
                                         noverlap=window_samples // 1.3)
    f_3_a, t_3_a, Sxx_3_a = signal.spectrogram(rir_no_rt[39:idx_pyroom], fs_compute, mode="angle",nperseg=window_samples,
                                         noverlap=window_samples // 1.3)
    #Convert into degrees

    Sxx_a = Sxx_a*180
    Sxx_1_a = Sxx_1_a*180
    Sxx_2_a = Sxx_2_a * 180
    Sxx_3_a = Sxx_3_a * 180

    #Calculate group delay

    Sxx_grp_delay=cal_grp_delay(t_a,f_a,Sxx_a)
    Sxx_1_grp_delay = cal_grp_delay(t_1_a, f_1_a, Sxx_1_a)
    Sxx_2_grp_delay = cal_grp_delay(t_2_a, f_2_a, Sxx_2_a)
    Sxx_3_grp_delay = cal_grp_delay(t_3_a, f_3_a, Sxx_3_a)


    #Calculate PSD

    f_p,pxx_rt=signal.welch(rir_rt,fs_compute,nperseg=window_samples)
    f_pno, pxx_nort = signal.welch(rir_no_rt, fs_compute, nperseg=window_samples)
    f_p_roomsim, pxx_roomsim = signal.welch(rir_roomsim, fs_compute, nperseg=window_samples)
    f_p_dechorate, pxx_dechorate = signal.welch(rir_dechorate, fs_compute, nperseg=window_samples)


    #Plot Graphs

    #Plot On Time Scale

    fig, axs = plt.subplots(4, 1, figsize=(30, 10))
    axs[0].plot(np.arange(sample_50ms), rir_rt[39:idx_pyroom], label="pyroom rt")

    axs[0].legend()

    axs[1].plot(np.arange(sample_50ms), rir_no_rt[39:idx_pyroom], label="pyroom no rt")
    axs[1].legend()

    axs[2].plot(np.arange(sample_50ms), rir_roomsim[:sample_50ms], label="roomsim rt")

    axs[2].legend()

    axs[3].plot(np.arange(sample_50ms), rir_dechorate[4619:idx_dechorate], label="dEchorate")

    print(np.argmax(rir_dechorate), np.argmax(rir_roomsim), np.argmax(rir_rt), np.argmax(rir_no_rt))

    # axs[3].legend()

    fig.tight_layout()

    plt.savefig("alignment_src_"+src_no+".png")

    plt.clf()

    fig,axs=plt.subplots(7,2,figsize=(12,20))

    mf = axs[0, 0].pcolormesh(t, f, 10 * np.log10(np.abs(Sxx)), shading='gouraud')
    axs[0, 0].set_xlabel("Time [sec]")
    axs[0, 0].set_ylabel("Frequency [Hz]")
    axs[0, 0].set_title("dEchorate")
    fig.colorbar(mf, use_gridspec=True, ax=axs[0, 0])

    g = axs[0, 1].pcolormesh(t_1, f_1, 10 * np.log10(np.abs(Sxx_1)), shading='gouraud')
    axs[0, 1].set_xlabel("Time [sec]")
    axs[0, 1].set_ylabel("Frequency [Hz]")
    axs[0, 1].set_title("Roomsim")
    fig.colorbar(g, use_gridspec=True, ax=axs[0, 1])

    k = axs[1, 0].pcolormesh(t_2, f_2, 10 * np.log10(np.abs(Sxx_2)), shading='gouraud')
    axs[1, 0].set_xlabel("Time [sec]")
    axs[1, 0].set_ylabel("Frequency [Hz]")
    axs[1, 0].set_title("Pyroom RT")
    fig.colorbar(k, use_gridspec=True, ax=axs[1, 0])

    l = axs[1, 1].pcolormesh(t_3, f_3, 10 * np.log10(np.abs(Sxx_3)), shading='gouraud')
    axs[1, 1].set_xlabel("Time [sec]")
    axs[1, 1].set_ylabel("Frequency [Hz]")
    axs[1, 1].set_title("Pyroom NORT")
    fig.colorbar(l, use_gridspec=True, ax=axs[1, 1])

    phase_1 = axs[2, 0].pcolormesh(t_a, f_a, 10 * np.log10(np.abs(Sxx_a)), shading='gouraud')
    axs[2, 0].set_xlabel("Time [sec]")
    axs[2, 0].set_ylabel("Frequency [Hz]")
    axs[2, 0].set_title("dEchorate")
    fig.colorbar(phase_1, use_gridspec=True, ax=axs[2, 0])

    phase_2 = axs[2, 1].pcolormesh(t_1_a, f_1_a, 10 * np.log10(np.abs(Sxx_1_a)), shading='gouraud')
    axs[2, 1].set_xlabel("Time [sec]")
    axs[2, 1].set_ylabel("Frequency [Hz]")
    axs[2, 1].set_title("Roomsim")
    fig.colorbar(phase_2, use_gridspec=True, ax=axs[2, 1])

    phase_3 = axs[3, 0].pcolormesh(t_2_a, f_2_a, 10 * np.log10(np.abs(Sxx_2_a)), shading='gouraud')
    axs[3, 0].set_xlabel("Time [sec]")
    axs[3, 0].set_ylabel("Frequency [Hz]")
    axs[3, 0].set_title("Pyroom RT")
    fig.colorbar(phase_3, use_gridspec=True, ax=axs[3, 0])

    phase_4 = axs[3, 1].pcolormesh(t_3_a, f_3_a, 10 * np.log10(np.abs(Sxx_3_a)), shading='gouraud')
    axs[3, 1].set_xlabel("Time [sec]")
    axs[3, 1].set_ylabel("Frequency [Hz]")
    axs[3, 1].set_title("Pyroom NORT")
    fig.colorbar(phase_4, use_gridspec=True, ax=axs[3, 1])

    grd_1 = axs[4, 0].pcolormesh(t_a, f_a, 10 * np.log10(np.abs(Sxx_grp_delay)), shading='gouraud')
    axs[4, 0].set_xlabel("Time [sec]")
    axs[4, 0].set_ylabel("Frequency [Hz]")
    axs[4, 0].set_title("dEchorate")
    fig.colorbar(grd_1, use_gridspec=True, ax=axs[4, 0])

    grd_2 = axs[4, 1].pcolormesh(t_1_a, f_1_a, 10 * np.log10(np.abs(Sxx_1_grp_delay)), shading='gouraud')
    axs[4, 1].set_xlabel("Time [sec]")
    axs[4, 1].set_ylabel("Frequency [Hz]")
    axs[4, 1].set_title("Roomsim")
    fig.colorbar(grd_2, use_gridspec=True, ax=axs[4, 1])

    grd_3 = axs[5, 0].pcolormesh(t_2_a, f_2_a, 10 * np.log10(np.abs(Sxx_2_grp_delay)), shading='gouraud')
    axs[5, 0].set_xlabel("Time [sec]")
    axs[5, 0].set_ylabel("Frequency [Hz]")
    axs[5, 0].set_title("Pyroom RT")
    fig.colorbar(grd_3, use_gridspec=True, ax=axs[5, 0])

    grd_4 = axs[5, 1].pcolormesh(t_3_a, f_3_a, 10 * np.log10(np.abs(Sxx_3_grp_delay)), shading='gouraud')
    axs[5, 1].set_xlabel("Time [sec]")
    axs[5, 1].set_ylabel("Frequency [Hz]")
    axs[5, 1].set_title("Pyroom NORT")
    fig.colorbar(grd_4, use_gridspec=True, ax=axs[5, 1])

    axs[6, 0].semilogy(f, pxx_rt, color="green", label="pyroom rt")
    axs[6, 0].semilogy(f, pxx_nort, color="blue", label="pyroom no_rt")
    axs[6, 0].semilogy(f, pxx_dechorate, color="orange", label="dEchorate")
    axs[6, 0].semilogy(f, pxx_roomsim, color="yellow", label="roomsim")

    axs[6, 0].set_ylabel("PSD [V**2/Hz]")
    axs[6, 0].set_xlabel("Frequency Hz")
    axs[6, 0].legend()

    fig.tight_layout(pad=2.0)

    plt.savefig("analysis_plots_src_"+src_no+".png", bbox_inches="tight")


