import numpy as np



abc=np.load("diego_data_mn_wm5_estimate_room_011000.npy")
abc_1=np.load("diego_data_mn_wm5_estimate_room_011100.npy")
abc_2=np.load("diego_data_mn_wm5_estimate_room_011110.npy")


std_vol=106.02265
std_surf=84.2240762
std_rt60=[ 0.7793691,0.7605436,0.6995225, 0.7076664, 0.6420753,0.51794204]
std_abs=[0.100825,0.1172430,0.1002776,0.09108845,0.09378748,0.091663032]


a1_rt1=np.abs((abc[1:,4]*std_rt60[2])-abc[1:,18]).std()
a1_rt2=np.abs((abc[1:,5]*std_rt60[3])-abc[1:,19]).std()
a1_rt3=np.abs((abc[1:,6]*std_rt60[4])-abc[1:,20]).std()
a1_rt4=np.abs((abc[1:,7]*std_rt60[5])-abc[1:,21]).std()

r_1=(a1_rt1+a1_rt2+a1_rt3+a1_rt4)/4

a2_rt1=np.abs((abc_1[1:,4]*std_rt60[2])-abc_1[1:,18]).std()
a2_rt2=np.abs((abc_1[1:,5]*std_rt60[3])-abc_1[1:,19]).std()
a2_rt3=np.abs((abc_1[1:,6]*std_rt60[4])-abc_1[1:,20]).std()
a2_rt4=np.abs((abc_1[1:,7]*std_rt60[5])-abc_1[1:,21]).std()


a2_rt1_st=(abc_1[1:,4]*std_rt60[2]).std()#-abc_1[1:,18]).mean()
a2_rt2_st=(abc_1[1:,5]*std_rt60[3]).std()#-abc_1[1:,19]).mean()
a2_rt3_st=(abc_1[1:,6]*std_rt60[4]).std()#-abc_1[1:,20]).mean()
a2_rt4_st=(abc_1[1:,7]*std_rt60[5]).std()#-abc_1[1:,21]).mean()

print("STD ROOM 011000",(a2_rt1_st+a2_rt2_st+a2_rt3_st+a2_rt4_st)/4)


r_2=(a2_rt1+a2_rt2+a2_rt3+a2_rt4)/4

a3_rt1=np.abs((abc_2[1:,4]*std_rt60[2])-abc_2[1:,18]).std()
a3_rt2=np.abs((abc_2[1:,5]*std_rt60[3])-abc_2[1:,19]).std()
a3_rt3=np.abs((abc_2[1:,6]*std_rt60[4])-abc_2[1:,20]).std()
a3_rt4=np.abs((abc_2[1:,7]*std_rt60[5])-abc_2[1:,21]).std()

r_3=(a2_rt1+a2_rt2+a2_rt3+a2_rt4)/4

print((r_1+r_2+r_3)/3)
print("=============")


a1_rt1=np.abs((abc[1:,10]*std_abs[2])-abc[1:,24]).std()
a1_rt2=np.abs((abc[1:,11]*std_abs[3])-abc[1:,25]).std()
a1_rt3=np.abs((abc[1:,12]*std_abs[4])-abc[1:,26]).std()
a1_rt4=np.abs((abc[1:,13]*std_abs[5])-abc[1:,27]).std()

k_1=(a1_rt1+a1_rt2+a1_rt3+a1_rt4)/4

a2_rt1=np.abs((abc_1[1:,10]*std_abs[2])-abc_1[1:,24]).std()
a2_rt2=np.abs((abc_1[1:,11]*std_abs[3])-abc_1[1:,25]).std()
a2_rt3=np.abs((abc_1[1:,12]*std_abs[4])-abc_1[1:,26]).std()
a2_rt4=np.abs((abc_1[1:,13]*std_abs[5])-abc_1[1:,27]).std()

a2_rt1_st=(abc_1[1:,10]*std_abs[2]).std()#-abc_1[1:,18]).mean()
a2_rt2_st=(abc_1[1:,11]*std_abs[3]).std()#-abc_1[1:,19]).mean()
a2_rt3_st=(abc_1[1:,12]*std_abs[4]).std()#-abc_1[1:,20]).mean()
a2_rt4_st=(abc_1[1:,13]*std_abs[5]).std()#-abc_1[1:,21]).mean()

print("STD ROOM 011000 AB",(a2_rt1_st+a2_rt2_st+a2_rt3_st+a2_rt4_st)/4)






k_2=(a2_rt1+a2_rt2+a2_rt3+a2_rt4)/4

a3_rt1=np.abs((abc_2[1:,10]*std_abs[2])-abc_2[1:,24]).std()
a3_rt2=np.abs((abc_2[1:,11]*std_abs[3])-abc_2[1:,25]).std()
a3_rt3=np.abs((abc_2[1:,12]*std_abs[4])-abc_2[1:,26]).std()
a3_rt4=np.abs((abc_2[1:,13]*std_abs[5])-abc_2[1:,27]).std()

k_3=(a2_rt1+a2_rt2+a2_rt3+a2_rt4)/4

print((k_1+k_2+k_3)/3)


print("================")


a1=np.abs(abc[1:,0]*std_surf-abc[1:,14]).std()
#print(a1)

a2=np.abs((abc_1[1:,0]*std_surf)-abc_1[1:,14]).std()
a2_st=(abc_1[1:,0]*std_surf).std()#-abc_1[1:,14]).mean()
print("STD ROOM 011100 surf",a2_st)

a3=np.abs((abc_2[1:,0]*std_surf)-abc_2[1:,14]).std()
#print(a3)
print((a1+a2+a3)/3)

print("========vol down=========")
a1=np.abs(abc[1:,1]*std_vol-abc[1:,15]).std() #.std()

#print(a1)

a2=np.abs((abc_1[1:,1]*std_vol)-abc_1[1:,15]).std()

a2_st=(abc_1[1:,1]*std_vol).std()#-abc_1[1:,14]).mean()
print("STD ROOM 011100 vol",a2_st)



#print(a2)
a3=np.abs((abc_2[1:,1]*std_vol)-abc_2[1:,15]).std()
#print(a3)
print((a1+a2+a3)/3)

