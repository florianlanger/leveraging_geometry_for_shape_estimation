
import os


import cv2
import torch
import numpy as np
from leveraging_geometry_for_shape_estimation.probabilistic_pose_and_shape.train_pose_classifier_from_lines.create_data.edge_detection.canny_edge_detector import CannyEdgeDetector
from tqdm import tqdm
# from scipy.misc import imsave


def main():
    device = torch.device("cuda:0")
    # for kind in ['val','train']:
    for kind in ['val_roca']:
        dir_path = '/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/scannet/data_03/{}/images_480_360/'.format(kind)
        out_path = '/scratch/fml35/datasets/own_datasets/leveraging_geometry_for_shape_estimation/scannet/data_03/{}/canny_480_360/'.format(kind)

        detector = CannyEdgeDetector(threshold=0.15).to(device)

        for file in tqdm(sorted(os.listdir(dir_path))):
            img = cv2.imread(os.path.join(dir_path, file))
            img = img.transpose((2, 0, 1))
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = torch.from_numpy(img).unsqueeze(0).to(device).float() / 255.

            output = detector(img)
            
            # for key in ['grad_magnitude','grad_orientation','thin_edges','thresholded_thin_edges']:
            for key in ['thresholded_thin_edges']:
                img = output[key].cpu().numpy()[0,0]
                img = ((img > 0.0) * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(out_path, file).replace('.jpg','.png'), img )


        # imsave('gradient_magnitude.png',grad_mag.data.cpu().numpy()[0,0])
        # imsave('thin_edges.png', thresholded.data.cpu().numpy()[0, 0])
        # imsave('final.png', (thresholded.data.cpu().numpy()[0, 0] > 0.0).astype(float))
        # imsave('thresholded.png', early_threshold.data.cpu().numpy()[0, 0])
        # print(output['thin_edges'].shape)
        # print(output['thresholded_thin_edges'].shape)

if __name__ == '__main__':
    with torch.no_grad():
        main()