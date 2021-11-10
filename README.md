# Leveraging Geometry for Shape Estimation from a Single RGB Image

Official implementation of the paper

<!-- > **Leveraging Geometry for Shape Estimation from a Single RGB Image** \
> BMVC 2021
> Florian Langer, [Ignas Budvytis][ignas], [Roberto Cipolla][roberto] \
> [[arXiv]][1] -->

Our proposed framework estimates shapes of objects in images by retrieving CAD models from a database and adapting and aligning them based on keypoint matches.

<div align="center">
  <img src="https://github.com/florianlanger/leveraging_geometry_for_shape_estimation/blob/main/assets/teaser.gif" width="550px" />
</div>


## Installation Requirements
- [Swin-Transformers][swin]
- [PyTorch3D][py3d]
- [OpenGV][ogv]
- [Blender][blend]

We recommend using a virtual environment.
```
python3.6 -m venv --system-site-packages ./venv
source ./venv/bin/activate
```

After insalling the packages above install additional dependencies by
```
python3.6 -m pip install -r requirements.txt

To install this repo
```
git clone https://github.com/florianlanger/leveraging_geometry_for_shape_estimation
cd leveraging_geometry_for_shape_estimation && pip install -e .
```

## Pipeline Overview
  1. Object Detection and Segmentation.
  Our object detection and segmentation is based on [Swin-Transformers][swin].
  2. CAD model retrieval.
  CAD models are rendered using [Blender][blend]. CAD model world coordinates are computed using [PyTorch3D][py3d].
  3. Keypoint Matching.
  Keypoint matching is performed using [SuperPoint][super].
  5. Pose Estimation.
  Pose estimation is performed using [OpenGV][ogv] and also [PyTorch3D][py3d].
  
  Depending on which steps of the pipeline you would like to modify you may not need to install all requirements listed above.


## Demo

Run our system on [Pix3D][pix].
1. Download Pix3D and replace the path in the ```global_config.json```
2. Change the path in ```model/model.sh``` to your Blender installation.
3. Run ```run_all.sh``` to run the whole pipeline.



## Citations
If you use this code please cite the following publication:
```
@inproceedings{langer_leveraging_shape,
               author = {Langer, F. and Budvytis, I. and Cipolla, R.},
               title = {Leveraging Geometry for Shape Estimation from a Single RGB Image},
               booktitle = {Proc. British Machine Vision Conference},
               month = {November},
               year = {2021},
               address={(Virtual)}
}
```

[1]: paper
[ogv]: https://laurentkneip.github.io/opengv/page_installation.html
[blend]: https://www.blender.org/download/
[py3d]: https://github.com/facebookresearch/pytorch3d
[pix]: https://github.com/xingyuansun/pix3d
[swin]: https://github.com/SwinTransformer/Swin-Transformer-Object-Detection
[super]: https://github.com/magicleap/SuperPointPretrainedNetwork
[roberto]: https://mi.eng.cam.ac.uk/~cipolla/
[ignas]: http://mi.eng.cam.ac.uk/~ib255/
