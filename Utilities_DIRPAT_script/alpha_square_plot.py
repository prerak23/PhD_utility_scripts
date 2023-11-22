import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

rir_roomsim=np.load("roomsim_alpha_check_48khz.npy")[0,:]
rir_pyroom=np.load("pyroom_alpha_check.npy")

sample_50ms=int((48000/1000)*50)
idx_pyroom=int(sample_50ms+39)

fig, axs = plt.subplots(3, 1, figsize=(30, 10))
axs[0].plot(np.arange(sample_50ms), rir_pyroom[39:idx_pyroom], label="pyroom rir")

axs[0].legend()

axs[1].plot(np.arange(sample_50ms), (rir_pyroom[39:idx_pyroom])**2, label="pyroom squared rir")
axs[1].legend()

axs[2].plot(np.arange(sample_50ms), rir_roomsim[:sample_50ms], label="roomsim rt")

axs[2].legend()


#print(np.argmax(rir_dechorate), np.argmax(rir_roomsim), np.argmax(rir_rt), np.argmax(rir_no_rt))

fig.tight_layout()

plt.savefig("alpha_square.png")
