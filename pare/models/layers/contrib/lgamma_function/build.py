from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

from torch import cuda
from torch.utils.ffi import create_extension

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Build tool')
    parser.add_argument('-with_cuda', default=False, action='store_true')
    args = parser.parse_args()
    with_cuda = args.with_cuda
    if with_cuda:
        print("COMPILING FOR CUDA!")
    else:
        print("COMPILING FOR CPU!")

    this_file = os.path.dirname(os.path.realpath(__file__))

    sources = ['src/specialmath.c']
    headers = ['src/specialmath.h']
    defines = []
    extra_objects = []

    if with_cuda:
        if not cuda.is_available():
            print('WARNING: Cuda not available!')
        else:
            print('INCLUDING CUDA CODE.')

            sources += ['src/specialmath_cuda.c']
            headers += ['src/specialmath_cuda.h']
            defines += [('WITH_CUDA', None)]
            with_cuda = True
            extra_objects = ['src/specialmath_cuda_impl.o']
            extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]

    ffi = create_extension(
        '_ext.specialmath',
        package=False,
        headers=headers,
        sources=sources,
        verbose=True,
        define_macros=defines,
        relative_to=__file__,
        with_cuda=with_cuda,
        extra_objects=extra_objects
    )

    ffi.build()
