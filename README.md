# SNVC

Official project website for the AAAI 2022 paper "Stereo Neural Vernier Caliper". This repository is under-preparation and the initial release is expected before the conference starts.

<p align="center">
  <img src="https://github.com/Nicholasli1995/SNVC/blob/main/imgs/teaser.png" height="210"/>
  <img src="https://github.com/Nicholasli1995/SNVC/blob/main/imgs/diagram.png" height="210"/>
</p>

TLDR: SNVC is a multi-resolutional voxel-based stereo 3D object detection approach. It features instance-level modeling and high-resolution local scene representation learning. It studies the local update problem in the stereo perception scenario, enabling model-agnostic refinement and tracking-by-detection.

## Demo: Detection
To be updated. SNVC constructs voxel-based global representaion for 3D object proposals, where the user is free to refine such proposals with an instance-level model.

## Demo: Instance tracking-by-detection
To be updated. Given an initial 3D cuboid, SNVC constructs a local 3D region-of-interest and searches the next position of the object.

## Demo: Model-agnostic instance-level refinement
To be updated. The instance-level model of SNVC can refine the predictions of different 3D object detectors.

## Data preparation
To be updated.

## Training 
To be updated.

## Inference
To be updated.

## TraceBox: a simple 3D bounding box visualization tool
This is a self-contained and simple python script for drawing 3D bounding boxes in an outdoor scene. It considers occlusion and uses simple ray tracing to obtain the visiblity of cuboid vertices (note the dashed lines). You can easily use this script in 3D object detection and 6DoF pose estimation projects.
<p align="center">
  <img src="https://github.com/Nicholasli1995/SNVC/blob/main/imgs/visualization.png" height="200"/>
</p>

## Environment
- Python 
- Numpy 
- PyTorch 
- CUDA

## License
This repository can be used for non-commercial purposes only. Contact me if you are interested in a commercial license. Third-party datasets like KITTI are subject to their own licenses and the user should obey them strictly.

## Citation
Please star this repository and cite the following paper in your publications if it helps your research:

    @inproceedings{li2022stereo,
      title={Stereo Neural Vernier Caliper},
      author={Li, Shichao and Liu, Zechun and Shen, Zhiqiang and Cheng, Kwang-Ting},
      booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
      volume={36},
      year={2022}
    }
    
Link to the paper:
To be updated
