wget -O leveraging_geometry_for_shape_estimation/keypoint_matching/demo_superpoint.py https://github.com/magicleap/SuperPointPretrainedNetwork/blob/master/demo_superpoint.py 

mkdir models
wget -O models/superpoint_v1.pth https://github.com/magicleap/SuperPointPretrainedNetwork/blob/master/superpoint_v1.pth

mkdir data
BASE=https://dl.fbaipublicfiles.com/meshrcnn
wget -O data/pix3d_s1_train.json $BASE/pix3d/pix3d_s1_train.json
wget -O data/pix3d_s1_test.json $BASE/pix3d/pix3d_s1_test.json
wget -O data/pix3d_s2_train.json $BASE/pix3d/pix3d_s2_train.json
wget -O data/pix3d_s2_test.json $BASE/pix3d/pix3d_s2_test.json