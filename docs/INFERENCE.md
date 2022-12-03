## Preparation
Before you start, please follow the instructions to prepare the dataset as described [here](https://github.com/Nicholasli1995/SNVC/blob/master/docs/DATASET.md). SNVC_DIR denotes the repository directory.

## Model-agnostic refinement
Download the model checkpoint and coarse proposals here.

Change directory to SNVC_DIR/tools and run

```bash
 python inference_agnostic.py --loadmodel CKPT_PATH --output_dir OUTPUT_DIR
```
Here CKPT_PATH is the path to the model checkpoint and OUTPUT_DIR is the directory storing output files.

This command loads IDA-3D predictions, refines the 3D object predictions corresponding to Table 3 in the paper.

To obtain quantitative evaluations, change directory to SNVC_DIR/tools/kitti-eval and run
```bash
 ./evaluate_object_3d_offline GT_DIR OUTPUT_DIR/all_parts
```
Here GT_DIR is the path to ground truth kitti labels.


