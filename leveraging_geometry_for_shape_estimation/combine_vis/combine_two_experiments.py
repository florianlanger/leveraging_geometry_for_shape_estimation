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


def main(exp1,exp2,exp3):

    dir1 = exp1 + '/combined_vis'
    dir2 = exp2 + '/combined_vis'
    dir3 = exp3 + '/combined_vis'

    outdir = exp1 + '/combined_vis_076_077'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    for name in tqdm(os.listdir(dir1)):
        img1 = load_image(dir1 + '/' + name,(512,1024))
        img2 = load_image(dir2 + '/' + name,(512,1024))
        img3 = load_image(dir3 + '/' + name,(512,1024))

        top = cv2.hconcat([img1,img2])
        bottom = cv2.hconcat([img3,img3*0])
        combined = cv2.vconcat([top,bottom])
        cv2.imwrite(outdir + '/' + name,combined)
       



if __name__ == '__main__':
    exp1 = sys.argv[1]
    exp2 = sys.argv[2]
    exp3 = sys.argv[3]
    main(exp1,exp2,exp3)