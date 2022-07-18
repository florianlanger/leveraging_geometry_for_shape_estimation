
import cv2
import os
import torch

# run this script on the machine that did run on with same environment

checkpoint_path = '/scratches/octopus/fml35/experiments/regress_T/runs_03_T_big/date_2022_05_26_time_19_04_11_three_refinements/saved_models/last_epoch.pth'

out_dir = '/scratches/octopus/fml35/experiments/regress_T/runs_03_T_big/date_2022_05_26_time_19_04_11_three_refinements/saved_models/epoch_255'
# os.mkdir(out_dir)

assert os.path.exists(checkpoint_path),'checkpoint path does not exist'
checkpoint = torch.load(checkpoint_path,map_location=torch.device('cpu'))

for key in checkpoint:
    torch.save(checkpoint[key],os.path.join(out_dir,key + '.pth'))
    print(key)


