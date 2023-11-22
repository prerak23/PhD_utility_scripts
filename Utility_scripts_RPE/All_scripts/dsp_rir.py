import matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import scipy
from scipy.io import loadmat
import numpy as np
from scipy import signal
import yaml
import random

def height_width_length():
    values=[random.randint(3,10),random.randint(3,10),round(random.uniform(2.5,4),1)]
    return values
def get_diffusion_coeff():
    coeff=round(random.uniform(0.2,1),2)
    return [coeff for x in range(36)]

def get_absorption_coeff():
    walls_profile = random.randint(0,1)
    celling_profile = random.randint(0,1)
    floor_profile = random.randint(0,1)
    if walls_profile:
        wall_value=round(random.uniform(0.01,0.12),2)
        four_wall_freq=[wall_value for x in range(6*4)]

    elif walls_profile == 0:
        wall_value_freq=[round(random.uniform(0.01,0.50),2),round(random.uniform(0.01,0.50),2),
                         round(random.uniform(0.01,0.30),2),round(random.uniform(0.01,0.12),2),
                         round(random.uniform(0.01,0.12),2),round(random.uniform(0.01,0.12),2)]
        four_wall_freq=([]+wall_value_freq)*4

    if celling_profile:
        celling_value = round(random.uniform(0.01, 0.12), 2)
        celling_freq = [celling_value for x in range(6)]
    elif celling_profile == 0:
        celling_freq = [round(random.uniform(0.01, 0.70), 2), round(random.uniform(0.15, 1.00), 2),
                           round(random.uniform(0.40, 1.00), 2), round(random.uniform(0.40, 1.00), 2),
                           round(random.uniform(0.40, 1.00), 2), round(random.uniform(0.30, 1.00), 2)]
    if floor_profile:
        floor_value = round(random.uniform(0.01, 0.12), 2)
        floor_freq = [floor_value for x in range(6)]
    elif floor_profile == 0:
        floor_freq =  [round(random.uniform(0.01, 0.20), 2), round(random.uniform(0.01, 0.30), 2),
                           round(random.uniform(0.05, 0.50), 2), round(random.uniform(0.15, 0.60), 2),
                           round(random.uniform(0.25, 0.75), 2), round(random.uniform(0.30, 0.80), 2)]
        return celling_freq+four_wall_freq+floor_freq

brir_roomsim=np.load("brir.npy")
print(brir_roomsim)
#plt.plot(np.arange(55125),brir_roomsim[0,:],linewidth=0.4)
#plt.show()
dict_file={}
humidity=0.42
temprature=20.0
reference_freq=125
for x in range(4):
    dict_file['room_'+str(x)]={'surface':{}}
    dict_file['room_'+str(x)]['dimension']=height_width_length()
    dict_file['room_'+str(x)]['humidity'] = 0.42
    dict_file['room_'+str(x)]['temprature'] = 20.0
    dict_file['room_'+str(x)]['surface']['frequency'] = [reference_freq*pow(2,a) for a in range(6)]
    dict_file['room_'+str(x)]['surface']['absorption'] = get_absorption_coeff()
    dict_file['room_' + str(x)]['surface']['diffusion'] = get_diffusion_coeff()
    #dict_file['room_'+str(x)]=room_dict['room_'+str(x)]
with open('conf_room_setup.yml', 'w') as file:
    documents = yaml.dump(dict_file, file)




'''
import random
sensors=["bidirectional","cardioid","dipole","hemisphere","hypercardioid","omnidirectional","subcardioid","supercardioid","unidirectional"]
def create_recv_sourc(ab:str):
    dict_file={}
    for x in range(2):
        dict_file[ab+'_'+str(x)]={'location':(8,2.5,3)}
        dict_file[ab+'_'+str(x)]['orientation']=(0,180,0)
        dict_file[ab+'_'+str(x)]['description']=random.choice(sensors)

    with open(ab+'.yml', 'w') as file:
        documents = yaml.dump(dict_file, file)

create_recv_sourc(ab='src')
create_recv_sourc(ab='rcv')










with open('conf_room_setup.yml','r') as file1:
    conf_room_setup=yaml.load(file1,Loader=yaml.FullLoader)
    print(conf_room_setup['room_0'])
'''

'''
y=loadmat('/home/psrivast/Téléchargements/roomsim/first_rir.mat')
x = loadmat('/home/psrivast/Téléchargements/roomsim/second_rir.mat')
print(x['r'].shape)
ot=y['r'][:,0]-x['r'][:,0]
fs=44100
f, t, Zxx=signal.stft(ot, fs, nperseg=10000)
print(np.abs(Zxx[1,:]))

plt.pcolormesh(t, f/1000, 20 * np.log10(np.abs(Zxx)), cmap='viridis')
plt.ylabel('Frequency [kHz]')
plt.xlabel('Time [s]');
#plt.savefig("diffuse_reflection_spect_1.jpeg")
plt.plot(np.arange(x['r'].shape[0]), ot, color='blue')
plt.show()
#plt.show()

'''