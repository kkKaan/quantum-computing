"""
gf2fast: High-Performance Linear Algebra over GF(2)
==================================================

A Python library for efficient binary matrix operations with optimized storage
and computation. Designed for applications in cryptography, coding theory,
network communications, and general-purpose binary linear algebra.

Key Features:
- Sparse matrix storage with multiple formats (CSR, bit-packed)
- Hardware-optimized bitwise operations
- Memory-efficient representations for structured matrices
- Complete linear algebra suite over GF(2)

Usage:
    >>> import gf2fast as gf2
    >>> # Create sparse binary matrix
    >>> from gf2fast import create_sparse_matrix
    >>> A = create_sparse_matrix(rows, cols, density=0.05)
    >>> # Solve linear system Ax = b over GF(2)
    >>> from gf2fast import solve
    >>> x = solve(A, b)
    >>> # Compute nullspace
    >>> from gf2fast import nullspace
    >>> null_vectors = nullspace(A)
"""

from .core import *
from .sparse import *
from .solvers import *
from .generators import *

__version__ = "0.1.0"
__author__ = "Optimized GF(2) Research"

# Export main classes and functions
__all__ = [
    # Core matrix classes
    'SparseGF2Matrix',
    'DenseGF2Matrix',

    # Basic operations
    'add',
    'multiply',
    'transpose',
    'rank',
    'det',
    'trace',
    'is_invertible',

    # Linear systems
    'solve',
    'nullspace',
    'inverse',
    'lu_decomposition',
    'least_squares',
    'kernel',
    'image',
    'rank_nullity_theorem',
    'solve_multiple_rhs',
    'iterative_refinement',
    'benchmark_solver',

    # Matrix generators
    'identity',
    'zeros',
    'ones',
    'random_sparse',
    'random_regular',
    'circulant',
    'circulant_random',
    'toeplitz',
    'vandermonde',
    'ldpc_matrix',
    # Quantum-specific generators intentionally not exported at top-level
    'hamming_matrix',
    'bch_matrix',

    # Advanced properties
    'reduced_row_echelon_form',
    'matrix_power',
    'characteristic_polynomial',
    'minimal_polynomial',
    'matrix_norm',
    'condition_number',

    # Factory
    'create_sparse_matrix'
]
