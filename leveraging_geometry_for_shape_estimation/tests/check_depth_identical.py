import numpy as np
from glob import glob


new_depths = '/data/cornucopia/fml35/experiments/test_output_all_1/models/depth/'
old_depths = '/data/cvfs/fml35/derivative_datasets/pix3d_new/own_data/depth/256/'
# new_depths = '/data/cvfs/fml35/derivative_datasets/pix3d_new/own_data/depth/dummy_test/'

for new_path in glob(new_depths + '*/*/*'):
    old_path = new_path.replace(new_depths,old_depths).replace('2.5.','2.0.').replace('7.5.','8.0.')
    print(old_path)

    old = np.load(old_path)
    new = np.load(new_path)
    # print(old[120:125,120:125])
    # print(new[120:125,120:125])
    # print(new[100:150,100:150])
    # print((new > 0).any())
    assert (np.abs(old - new) < 0.005).all()
print('done')