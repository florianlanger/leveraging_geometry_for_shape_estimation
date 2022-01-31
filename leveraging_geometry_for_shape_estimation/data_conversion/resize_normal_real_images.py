import json
import cv2
from tqdm import tqdm

source_dir = '/scratches/gwangban_3/fml35/exp/02_pix3d_pred/'

target_dir = '/scratch/fml35/datasets/pix3d_new/own_data/real_images_3d/predicted_normals/'
pix_path = '/scratch/fml35/datasets/pix3d_new/pix3d.json'
with open(pix_path,'r') as f:
    pix3d = json.load(f)

for data in tqdm(pix3d):
    read_path = source_dir + data["category"] + '/results/' + data['img'].split('/')[2].split('.')[0] + '_pred_norm.png'
    write_path = target_dir + data["category"] + '/' + data['img'].split('/')[2].split('.')[0] + '.png'

    img = cv2.imread(read_path)
    resized = cv2.resize(img,tuple(data["img_size"]))
    cv2.imwrite(write_path,resized)