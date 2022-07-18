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

def load_image_potentially_different_orientation(path,size,file_exists=False):
    img_exist = False
    for orientation in range(4):
        add_on = str(orientation).zfill(2)
        check_path = path.rsplit('_',1)[0] + '_' + add_on + '.png'
        img = load_image(check_path,size)
        if os.path.exists(check_path):
            img_exist = True
            break
    if file_exists == True:
        assert img_exist,path

    return img

def load_image_potentially_different_orientation_category_ending(path,size,file_exists=False):
    img_exist = False
    for orientation in range(4):
        add_on = str(orientation).zfill(2)
        check_path = path.rsplit('_',2)[0] + '_' + add_on + '_' + path.split('_')[-1] #+'.png'
        img = load_image(check_path,size)
        if os.path.exists(check_path):
            img_exist = True
            break
    if file_exists == True:
        assert img_exist,path

    return img

def main(exp1,exp2,exp3,exp4):

    # dir1 = exp1 + '/combined_vis'
    # dir2 = exp2 + '/combined_vis'
    # dir3 = exp3 + '/combined_vis'
    # dir4 = exp4 + '/combined_vis'

    # dir1 = exp1 + '/poses_vis'
    # dir2 = exp2 + '/poses_vis'
    # dir3 = exp3 + '/poses_vis'
    dir4 = exp4 + '/poses_vis'

    dir1 = exp1 + '/T_lines_vis_new_model_all_gt'
    dir2 = exp1 + '/T_lines_vis_new_model_filtered_gt'
    dir3 = exp1 + '/T_lines_vis_old_model_gt'
    dir5 = exp1 + '/T_lines_vis_lines_gt_objects_vis_1_overlayed'
    # dir4 = exp1 + '/non'

    check_all_exist = False

    outdir = exp1 + '/T_lines_vis_combined_3_new_model_all_gt_new_model_filtered_gt_old_model_gt'
    print(outdir)
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    for name in tqdm(sorted(os.listdir(dir1))):
        img_size = (960,1280)
        # img_size = (512,1024)
        img1 = load_image(dir1 + '/' + name,img_size)

        # img2 = load_image(dir2 + '/' + name,(512,1024))
        # img3 = load_image(dir3 + '/' + name,(512,1024))
        # img4 = load_image(dir4 + '/' + name,(512,1024))
        # assert os.path.exists(dir2 + '/' + name),dir2 + '/' + name
        # img2 = load_image_potentially_different_orientation(dir2 + '/' + name,img_size,file_exists=check_all_exist)
        # img3 = load_image_potentially_different_orientation(dir3 + '/' + name,img_size,file_exists=check_all_exist)
        # img4 = load_image_potentially_different_orientation(dir4 + '/' + name,img_size,file_exists=check_all_exist)

        img2 = load_image_potentially_different_orientation(dir2 + '/' + name,img_size,file_exists=check_all_exist)
        img3 = load_image_potentially_different_orientation(dir3 + '/' + name,img_size,file_exists=check_all_exist)
        # img4 = load_image_potentially_different_orientation_category_ending(dir4 + '/' + name,img_size,file_exists=check_all_exist)

        # print(dir4 + '/' + name.rsplit('_',1)[0] +'.png')
        # assert os.path.exists(dir4 + '/' + name.rsplit('_',1)[0] +'_00.png'),dir4 + '/' + name.rsplit('_',1)[0] +'_00.png'
        img4 = load_image_potentially_different_orientation(dir4 + '/' + name.rsplit('_',1)[0] +'_00.png',img_size,file_exists=check_all_exist)
        img5 = load_image_potentially_different_orientation(dir5 + '/' + name,img_size,file_exists=check_all_exist)
        

        top = cv2.hconcat([img1,img2,img5])
        bottom = cv2.hconcat([img3,img4,img5*0])
        combined = cv2.vconcat([top,bottom])
        cv2.imwrite(outdir + '/' + name,combined)

if __name__ == '__main__':
    # exp1 = sys.argv[1]
    # exp2 = sys.argv[2]
    # exp3 = sys.argv[3]
    # exp4 = sys.argv[4]
    # exp1 = '/scratch2/fml35/experiments/leveraging_geometry_for_shape/exp_186_roca_retrieval_gt_z_lines_octopus'
    # exp2 = '/scratch2/fml35/experiments/leveraging_geometry_for_shape/exp_178_gt_retrieval_matches_gt_z'
    # exp3 = '/scratch2/fml35/experiments/leveraging_geometry_for_shape/exp_156_roca_all_vis'
    exp4 = '/scratch2/fml35/experiments/leveraging_geometry_for_shape/exp_157_all_vis_gt'

    exp1 = '/scratch2/fml35/experiments/leveraging_geometry_for_shape/exp_203_new_lines_from_2d_all_vis'
    exp2 = None
    exp3 = None
    # exp4 = None
    main(exp1,exp2,exp3,exp4)