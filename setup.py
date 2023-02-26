import os
import subprocess
import torch

from os.path import join as osp
from setuptools import find_packages
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension


def get_git_commit_number():
    if not os.path.exists('.git'):
        return '0000000'

    cmd_out = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE)
    git_commit_number = cmd_out.stdout.decode('utf-8')[:7]
    return git_commit_number

def write_version_to_file(version, target_file):
    with open(target_file, 'w') as f:
        print('__version__ = "{:s}"'.format(version), file=f)
        
def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = osp(this_dir, "snvc", "extension")
    osp(extensions_dir, 'iou3d_nms', 'src', 'iou3d_cpu.cpp')
    ext_modules=[
        CUDAExtension(
            name='snvc.extension.iou3d_nms.iou3d_nms_cuda',
            sources=[
                osp(extensions_dir, 'iou3d_nms', 'src', 'iou3d_cpu.cpp'),
                osp(extensions_dir, 'iou3d_nms', 'src', 'iou3d_nms_api.cpp'),
                osp(extensions_dir, 'iou3d_nms', 'src', 'iou3d_nms.cpp'),
                osp(extensions_dir, 'iou3d_nms', 'src', 'iou3d_nms_kernel.cu')
            ],         
        ),
        CUDAExtension(
            name='snvc.extension.build_cost_volume.build_cost_volume_cuda',
            sources=[
                osp(extensions_dir, 'build_cost_volume', 'src', 'BuildCostVolume.cpp'),
                osp(extensions_dir, 'build_cost_volume', 'src', 'BuildCostVolume_cuda.cu')
            ],
            define_macros=[("WITH_CUDA", None)]
        ),
        CUDAExtension(
            name='snvc.extension.roiaware_pool3d.roiaware_pool3d_cuda',
            sources=[
                osp(extensions_dir, 'roiaware_pool3d', 'src', 'roiaware_pool3d.cpp'),
                osp(extensions_dir, 'roiaware_pool3d', 'src', 'roiaware_pool3d_kernel.cu'),                
            ]
        ),
    ]
        
    return ext_modules

if __name__ == '__main__':
    version = '0.9.0+{:s}'.format(get_git_commit_number())
    write_version_to_file(version, 'snvc/version.py')
    setup(
        name="snvc",
        version=version,
        author="Shichao Li",
        author_email="nicholas.li@connect.ust.hk",
        license="MIT License",        
        url="https://github.com/Nicholasli1995/SNVC",
        description="A python implementation of Stereo Neural Vernier Caliper.",
        install_requires=[
            'numpy',
            'torch',
            'numba',
        ],        
        packages=find_packages(exclude=("configs", 
                                        "tools", 
                                        "data", 
                                        "docs", 
                                        "imgs")),
        cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
        ext_modules=get_extensions()        
    )