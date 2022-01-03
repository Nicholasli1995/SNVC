## Data Preparation
The KITTI object detection benchmark is used in this study. If you want to use your own dataset, you need to prepare your image, calibration, and cuboid labels accordingly.
 
You can download the KITTI dataset [here](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d). You need to download left/right color images, calibration files, and object labels.

You can download the split files [here](https://drive.google.com/drive/folders/1YLtptqspOFw08QG2MsxewDT9tjF2O45g?usp=sharing) and place them at ${YOUR_KITTI_DIR}/SPLIT/ImageSets.
Your data folder should look like this:

   ```
   ${YOUR_KITTI_DIR}
   ├── training
      ├── calib
          ├── xxxxxx.txt (Camera parameters for image xxxxxx)
      ├── image_2
          ├── xxxxxx.png (left color image xxxxxx)
      ├── image_3
          ├── xxxxxx.png (right color image xxxxxx)
      ├── label_2
          ├── xxxxxx.txt (object labels for image xxxxxx)
      ├── ImageSets
         ├── train.txt
         ├── val.txt   
         ├── trainval.txt        
   ├── testing
      ├── calib
          ├── xxxxxx.txt (Camera parameters for image xxxxxx)
      ├── image_2
          ├── xxxxxx.png (left color image xxxxxx)
      ├── image_3
          ├── xxxxxx.png (right color image xxxxxx)
      ├── ImageSets
         ├── test.txt
   ```
