import json
from scipy.spatial.transform import Rotation as scipy_rot
import numpy as np


N = 1000

all_data = []

for i in range(N):

    w = np.random.randint(300,1000)
    h = np.random.randint(300,1000)
    f = 20 + np.random.rand() * 20
    T = (np.random.rand(3)-0.5) * np.array([0.2,0.2,0.2]) + np.array([0.0,0.0,1.4])
    R = scipy_rot.random().as_matrix()
    # img_name = 'img_{}.png'.format(str(i).zfill(3))
    # model = 'model_{}.obj'.format(str(int(np.floor(i/2))).zfill(3))
    img_name = 'img_{}.png'.format(str(i).zfill(6))
    model = 'model_{}.obj'.format(str(int(np.floor(i))).zfill(6))
    euler_zyx = scipy_rot.from_matrix(R).as_euler('zyx')

    data = {"img": img_name,"w": w, "h": h,"f":f,"model": model,"trans_mat": T.tolist(),"rot_mat": R.tolist(),'euler_zyx': euler_zyx.tolist()}
    all_data.append(data)


target_path = '/scratch/fml35/datasets/cubes_01_large/img_info.json'
with open(target_path,'w') as f:
    json.dump(all_data,f,indent=4)
    