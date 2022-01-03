# SNVC

Official project website for the AAAI 2022 paper "Stereo Neural Vernier Caliper". 

Note: this repository is still under-preparation and the initial release is expected before the conference starts.

<p align="center">
  <img src="https://github.com/Nicholasli1995/SNVC/blob/main/imgs/teaser.png" height="210"/>
</p>

TLDR: SNVC is a multi-resolution voxel-based stereo 3D object detection (S3DOD) approach. It features **object-centric** modeling and is capable of learning high-resolution **local** scene representations for precise object center estimation. It studies the **local update problem** in the stereo perception scenario, enabling model-agnostic refinement and tracking-by-detection.

## Environment
Refer to ENV.md to build this project.

## Demo: Detection
https://user-images.githubusercontent.com/34362048/147870110-398bde6a-4832-47e2-861f-011788023402.mp4

Refer to DEMO_DET.md to see how to perform frame-by-frame S3DOD with an examplar python script. SNVC constructs voxel-based global representaion for 3D object proposals, where the user is free to refine such proposals with an instance-level model. If you want to generate a video like this, check out the visualization tool in this repo.

## Demo: Instance tracking-by-detection
Refer to DEMO_TBD.md for an examplar tracking-by-detection script. Given an initial 3D cuboid prediction, SNVC is capable of constructing a high-resolution local 3D region-of-interest and searches the next position of the object.

## Demo: Model-agnostic instance-level refinement
Refer to DEMO_REFINE.md for a script demonstrating how the instance-level modeling capability in this repo can be used with other detectors. The instance-level model of SNVC can refine the predictions of different 3D object detectors in a model-agnostic mannner. This means you can use **any** real-time 3D object detector to produce coarse proposals and refine them only when necessary.
<p align="center">
<img src="https://github.com/Nicholasli1995/SNVC/blob/main/imgs/diagram.png" height="210"/>
</p>

## Data preparation
Refer to DATASET.md to download and prepare the KITTI dataset used in this project.

## Training 
Refer to TRAIN.md to train the models by yourself.

## Inference
Refer to INFERENCE.md to use pre-trained weights and re-produce the quantitative results.

## TraceBox: a simple 3D bounding box visualization tool
This is a self-contained and simple python script for drawing a set of 3D cuboids. It considers occlusion and uses simple ray tracing to obtain the visiblity of cuboid vertices (note the dashed lines). You can easily use this script in 3D object detection and 6DoF pose estimation projects.
<p align="center">
  <img src="https://github.com/Nicholasli1995/SNVC/blob/main/imgs/visualization.png" height="200"/>
</p>

## License
This repository can be used for non-commercial purposes only. Contact me (nicholas.li@connect.ust.hk) if you are interested in a commercial license. Third-party datasets like KITTI are subject to their own licenses and the user should obey them strictly.

## Citation
Please star this repository and cite the following paper in your publications if it helps your research:

    @inproceedings{li2022stereo,
      title={Stereo Neural Vernier Caliper},
      author={Li, Shichao and Liu, Zechun and Shen, Zhiqiang and Cheng, Kwang-Ting},
      booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
      volume={36},
      year={2022}
    }
    
Link to the paper: to be updated

## Acknowledgement
Certain modules were adapted from DSGN and LIGA-Stereo and thank the original authors for their contributions.
