import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 

sns.set_theme()





path="/home/psrivastava/baseline/scripts/pre_procesing/z_test/real_data_results/"
files_=os.listdir(path)
for r in file_:
    tmp_=r.split("_")[-1]
    room_no=tmp_.split(".")[0]
    abc=np.load(path+r)[1:,:]
    
    
