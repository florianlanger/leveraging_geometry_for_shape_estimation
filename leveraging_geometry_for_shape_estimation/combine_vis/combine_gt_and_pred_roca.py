import os
from tqdm import tqdm
import cv2
from combine_vis import load_image
import sys


# def main(exp1,exp2):


#     dir1 = exp1 + '/combined_vis'
#     dir2 = exp2 + '/combined_vis'

#     outdir = exp1 + '/combined_vis_' + exp2.split('/')[-1]
#     if not os.path.exists(outdir):
#         os.mkdir(outdir)
#     for name in tqdm(os.listdir(dir1)):
#         img1 = load_image(dir1 + '/' + name,(512,1024))
#         img2 = load_image(dir2 + '/' + name,(512,1024))

#         combined = cv2.vconcat([img1,img2])
#         cv2.imwrite(outdir + '/' + name,combined)


def main(exp_pred,exp_gt):

    dir1 = exp_pred + 'poses_vis'
    dir2 = exp_pred + 'segmentation_vis'
    dir3 = exp_gt + 'poses_vis'

    outdir = exp_pred + '/combined_vis_gt'

    for name in tqdm(os.listdir(dir1)):
        img1 = load_image(dir1 + '/' + name,(600,800))
        # print(dir2 + '/' + name.rsplit('_',2)[0] + '.png')
        img2 = load_image(dir2 + '/' + name.rsplit('_',2)[0] + '.png',(600,800))
        img3 = load_image(dir3 + '/' + name,(600,800))

        top = cv2.hconcat([img2,img1,img3])
        cv2.imwrite(outdir + '/' + name,top)
       



if __name__ == '__main__':
    # exp1 = sys.argv[1]
    # exp2 = sys.argv[2]
    # exp3 = sys.argv[3]
    exp_pred = '/scratch2/fml35/experiments/leveraging_geometry_for_shape/exp_144_eval_roca/'
    exp_gt = '/scratch2/fml35/experiments/leveraging_geometry_for_shape/exp_145_eval_gt_roca/'
    main(exp_pred,exp_gt)