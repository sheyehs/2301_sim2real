from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os 
os.environ['TORCH_CUDA_ARCH_LIST'] = "3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"

setup(
    name='ransac_voting_3d',
    ext_modules=[
        CUDAExtension('ransac_voting_3d', [
            './src/ransac_voting.cpp',
            './src/ransac_voting_kernel.cu'
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
