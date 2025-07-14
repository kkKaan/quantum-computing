"""
gf2fast: High-Performance Linear Algebra over GF(2)
==================================================

A Python library for efficient binary matrix operations with optimized storage
and computation. Designed for applications in quantum error correction,
cryptography, coding theory, and network communications.

Key Features:
- Sparse matrix storage with multiple formats (CSR, CSC, bit-packed)
- Hardware-optimized bitwise operations
- Memory-efficient representations for structured matrices
- Complete linear algebra suite over GF(2)
- Specialized functions for quantum error correction and coding theory

Usage:
    >>> import gf2fast as gf2
    >>> # Create sparse binary matrix
    >>> A = gf2.sparse_matrix(rows, cols, density=0.05)
    >>> # Solve linear system Ax = b over GF(2)
    >>> x = gf2.solve(A, b)
    >>> # Compute nullspace
    >>> null_vectors = gf2.nullspace(A)
"""

from .core import *
from .sparse import *
from .solvers import *
from .generators import *
from .quantum import *

__version__ = "0.1.0"
__author__ = "Optimized GF(2) Research"

# Export main classes and functions
__all__ = [
    # Core matrix classes
    'SparseGF2Matrix',
    'DenseGF2Matrix',
    'BitPackedMatrix',

    # Basic operations
    'add',
    'multiply',
    'transpose',
    'rank',
    'det',

    # Linear systems
    'solve',
    'nullspace',
    'inverse',
    'lu_decomposition',

    # Matrix generators
    'identity',
    'random_sparse',
    'circulant',
    'vandermonde',
    'ldpc_matrix',
    'surface_code_matrix',
    'hamming_matrix',

    # Quantum computing
    'pauli_group',
    'stabilizer_matrix',
    'syndrome_decode',

    # Utilities
    'benchmark',
    'memory_usage',
    'sparsity_stats'
]
