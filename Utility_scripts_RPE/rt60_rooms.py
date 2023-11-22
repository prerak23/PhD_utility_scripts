import numpy as np
import h5py 

abc=h5py.File("/home/psrivastava/baseline/scripts/pre_processing/rt60_anno_room_20_median.hdf5","r")
room_test=np.arange(18000,20000)
rooms_less_reverb=[]
rooms_med_reverb=[]
rooms_high_reverb=[]
for r in room_test:
    #print(r)
    med_rt60=abc["room_nos"]["room_"+str(r)]["rt60"][()]
    #print(med_rt60)
    if (med_rt60<0.5).sum() == 6:
        rooms_less_reverb.append(r)
        print("less reverb")
        print(med_rt60)
    elif (med_rt60>0.5).sum() == 6:
        if (med_rt60<1.5).sum() == 6:
            print("med reverb")
            print(med_rt60)
            rooms_med_reverb.append(r)
        elif (med_rt60 > 1.5).sum() == 6:
            rooms_high_reverb.append(r)
            print("high reverb")
            print(med_rt60)
print(len(rooms_less_reverb))
print(len(rooms_high_reverb))
print(len(rooms_med_reverb))
np.save("less_reverb_rooms.npy",rooms_less_reverb)
np.save("mid_reverb_rooms.npy",rooms_med_reverb)
np.save("high_reverb_rooms.npy",rooms_high_reverb)

