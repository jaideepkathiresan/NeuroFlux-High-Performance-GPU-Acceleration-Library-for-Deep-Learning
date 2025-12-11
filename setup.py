from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

sources = [
    'src/cpp/bindings.cpp',
    'src/cuda/gemm_kernel.cu',
    'src/cuda/fused_ops.cu',
    'src/cuda/wmma_gemm.cu',
    'src/cuda/flash_attn.cu',
    'src/cuda/layernorm.cu'
]

# Compiler flags for high performance
# Note: wmma requires gpu-arch flags, e.g., -gencode arch=compute_75,code=sm_75
nvcc_args = [
    '-O3', 
    '-U__CUDA_NO_HALF_OPERATORS__', 
    '-U__CUDA_NO_HALF_CONVERSIONS__',
    '-U__CUDA_NO_HALF2_OPERATORS__',
    '--use_fast_math'
]

setup(
    name='neuroflux_cpp',
    version='1.0.0',
    ext_modules=[
        CUDAExtension(
            name='neuroflux_cpp',
            sources=sources,
            include_dirs=[os.path.join(os.getcwd(), 'src/include')],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': nvcc_args}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    description='High-Performance Deep Learning Optimization Library'
)
