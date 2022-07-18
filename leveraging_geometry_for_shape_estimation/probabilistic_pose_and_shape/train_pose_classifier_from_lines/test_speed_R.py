from scipy.spatial.transform import Rotation as scipy_rot

from time import time
t1 = time()
a = scipy_rot.from_quat([1,0.4,1,0.3])
b = a.as_matrix()
t2 = time()
print(t2-t1)