# Leveraging Geometry for Shape Estimation

Code for the paper

**[Leveraging Geometry for Shape Estimation from a Single RGB Image][1]**  
Florian Langer, Ignas Budvytis, Roberto Cipolla
BMVC 2021

<div align="center">
  <img src="https://www.youtube.com/watch?v=n6WD9km8zRQ" width="550px" />
</div>


## Installation Requirements
- [Swin-Transformers][swin]
- [PyTorch3D][py3d]
- [OpenGV][ogv]
- [Blender][blend]

To install this repo
```
git clone https://github.com/florianlanger/leveraging_geometry_for_shape_estimation
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
