"""
Code adapted from https://github.com/davidcpage/seqdist
"""
import cupy as cp
from pathlib import Path
import numpy as np

def add_checks(raw_kernel):
    def wrapped(grid, block, args, *, shared_mem=0):
        MAX_THREADS = 1024
        if np.prod(block) > MAX_THREADS:
            raise Exception('Block of size {} not allowed. Maximum number of threads allowed per block is {}.'.format(block, MAX_THREADS))
        return raw_kernel(grid, block, args, shared_mem=shared_mem)
    return wrapped


def load_cupy_func(fname, name, **kwargs):
    try: fname = str((Path(__file__).parent / fname).resolve())
    except: pass
    with open(fname) as f:
        code = f.read()
    macros = ['#define {!s} {!s}'.format(k, v) for k,v in kwargs.items()]
    code = '\n'.join(macros + [code])
    return add_checks(cp.RawKernel(code, name))
