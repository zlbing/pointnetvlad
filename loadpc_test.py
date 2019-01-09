import os
import numpy as np
from numpy import linalg as LA
BASE_PATH = "/home/zzz/work-space/benchmark_datasets"
filename = "oxford/2015-10-30-13-52-14/pointcloud_20m/1446213237183052.bin"
#filename = "kaicheng/2018-12-12-10-17-39/pointcloud/1544581256982785000.bin"
pc=np.fromfile(os.path.join(BASE_PATH,filename), dtype=np.float64)
pc=np.reshape(pc,(pc.shape[0]//3,3))
print("pc.norm",pc[0,2],LA.norm(pc[0]))
print(pc)

##scale points to [-1,1]
max_value = [60]
min_value = [-60]
pc=np.reshape(pc,(pc.shape[0]*pc.shape[1]))
print("pc.shape",pc.shape)
# max_value.append(max(pc))
# min_value.append(min(pc))
for k in range(len(pc)):
    pc[k]=((pc[k]-min_value[0])/(max_value[0] - min_value[0]))*2 -1

print("max_value=",max_value,"min_value=",min_value)
pc=np.reshape(pc,(pc.shape[0]//3,3))
print("pc.norm",pc[0,2],LA.norm(pc[0]))
print(pc)