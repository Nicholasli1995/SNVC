## Environment
You need to create an environment that meets the following dependencies. The following are two examples of environments. The versions included in the parenthesis are **tested**. Other versions may also work but are **not tested**.

A tested local PC with NVIDIA TITAN Xp GPUs (2*12GB, driver 396.26):

- Python (3.7.9)
- Numpy (1.19.2)
- PyTorch (1.7.1, GPU required)
- Scipy (1.5.4)
- Matplotlib (3.3.3)
- OpenCV-python (4.4.0.46)
- CUDA (9.2.88)

Another tested computing node with NVIDIA RTX 3090 GPUs (8*24 GB, driver 470.86):

- Python (3.7.9)
- Numpy (1.20.3)
- PyTorch (1.9.0, GPU required)
- Scipy (1.6.2)
- Matplotlib (3.4.2)
- OpenCV (3.4.2)
- CUDA (11.4)

For more details of the environment, you can refer to the spec-file.txt. 

The recommended environment manager is [Anaconda](https://www.anaconda.com/), which can create an environment using this provided spec-list. For debugging using an IDE, the recommended IDE is Spyder which you can get by
```bash
conda install spyder
```

## Compile and install
Compile and install the extension modules at your project directory ${SNVC_DIR} with:
```bash
python setup.py develop 
```

## Compile the KITTI evaluator (recommended)
If you need to quantitatively evaluate the model predictions for KITTI, you need to compile this evaluation tool.
```bash
cd ${SNVC_DIR}/tools/kitti-eval 
```
Compile the source code
```bash
g++ -o evaluate_object_3d_offline evaluate_object_3d_offline.cpp -O3
```
