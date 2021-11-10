# Leveraging Geometry for Shape Estimation

Code for the paper

**[Leveraging Geometry for Shape Estimation from a Single RGB Image][1]**  
Florian Langer, [Ignas Budvytis][ignas], [Roberto Cipolla][roberto]
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

## Config

There are a few fields in ```global_config.json``` that you have to adjust change:
- `general/target_folder`: output_directory
- `general/models_folder_read`: output_directory
- `general/image_folder`: output_directory + `/images`
- `general/mask_folder`: output_directory + `/masks`
- `dataset/pix3d_path`: path to the downloaded pix3d directory.

Additionally there are some fields which you may want to change to run different configurations. These include

- `segmentation/use_gt`: whether to use ground truth masks or not. This must be either `"True"` or `"False"`. If set to `"True"` values for `segmentation/config` and `segmentation/checkpoint` are not accessed.
- `segmentation/config` and `segmentation/checkpoint`: Path to segmentation config and path to segmentation checkpoint. These have to match. See `models` and `configs` for available netorks. Also note that these should correspond to the split that is set in `dataset/split`
- `retrieval/checkpoint_file`: path to embedding network used for retrieval. Note that these should match the split in `dataset/split` and `segmentation/use_gt`




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
