import h5py
import numpy as np
abc=h5py.File("/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/psrivast/snr_histogram_check.hdf5","r")
test_rooms=np.arange(18000,20000)
vps=np.arange(1,6)
points_below_zero=[]
points_above_zero=[]


for r in test_rooms:
    print(r)
    snr_f=abc["room_nos"]["room_"+str(r)]["nsmix_snr_f"][()]
    snr_diff=abc["room_nos"]["room_"+str(r)]["nsmix_snr_f"][()]
    snr_f=np.array(snr_f)
    snr_diff=np.array(snr_diff)
    print(snr_f)
    vp_greater_zero=np.nonzero(snr_f>0)[0]
    print(vp_greater_zero)
    vp_below_zero=np.nonzero(snr_f<0)[0]
    if vp_greater_zero.size > 0:
        for i in vp_greater_zero:
            points_above_zero.append(["room_"+str(r),str(i+1)])
    if vp_below_zero.size > 0:
        for i in vp_below_zero:
            points_below_zero.append(["room_"+str(r),str(i+1)])

np.save("test_set_snr_below_zero",np.array(points_below_zero))
np.save("test_set_snr_above_zero",np.array(points_above_zero))




