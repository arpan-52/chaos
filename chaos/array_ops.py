"""
Array operations abstraction: auto-detect CuPy, fallback to NumPy.

Usage:
    from .array_ops import xp, using_gpu, to_cpu, to_gpu
"""

# Try to import CuPy, fallback to NumPy
try:
    import cupy as xp
    using_gpu = True
    print("[CHAOS] GPU acceleration enabled (CuPy detected)")
except ImportError:
    import numpy as xp
    using_gpu = False
    print("[CHAOS] Using CPU (NumPy) - install CuPy for GPU acceleration")


def to_cpu(arr):
    """Convert array to NumPy (CPU), handling both CuPy and NumPy."""
    if using_gpu and hasattr(arr, 'get'):
        return arr.get()
    return arr


def to_gpu(arr):
    """Convert array to CuPy (GPU) if available, else return as-is."""
    if using_gpu and not hasattr(arr, 'get'):
        return xp.asarray(arr)
    return arr


__all__ = ['xp', 'using_gpu', 'to_cpu', 'to_gpu']
