from tqdm import tqdm
import cv2
import os
import numpy as np

from pytorch3d.ops import sample_farthest_points

import torch



def main():

    n_points = 5000
    device = torch.device("cuda:0")

    target_dir = '/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/scannet/data_03/'
    kind = 'val_roca'
    target_dir += kind

    input_dir = target_dir + '/canny_480_360'
    output_dir = target_dir + '/canny_points_480_360_n_points_' + str(n_points)
    output_dir_vis = target_dir + '/canny_points_vis_480_360_n_points_' + str(n_points)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(output_dir_vis):
        os.mkdir(output_dir_vis)


    for img_name in tqdm(sorted(os.listdir(input_dir))):
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        indices = np.nonzero(img)
        indices = np.stack(indices,axis=1)

        resampled,_ = sample_farthest_points(torch.Tensor(indices).unsqueeze(0).to(device), K=n_points)

        resampled = resampled[0].cpu().numpy().astype(np.int16)

        np.save(os.path.join(output_dir, img_name.split('.')[0]), resampled)

        img_new = img * 0
        img_new[resampled[:,0],resampled[:,1]] = img_new[resampled[:,0],resampled[:,1]] * 0 + 255
        cv2.imwrite(os.path.join(output_dir_vis, img_name.split('.')[0] + '.png'), img_new)



if __name__ == '__main__':
    with torch.no_grad():
        main()