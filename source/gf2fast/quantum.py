"""
Quantum Computing Utilities for GF(2) Operations
===============================================

Specialized functions for quantum error correction, stabilizer computations,
and quantum code analysis using optimized GF(2) operations.

Functions:
- Pauli group operations
- Stabilizer arithmetic
- Syndrome decoding for quantum codes
- Quantum code analysis
"""

import numpy as np
from typing import List, Tuple, Optional, Union
from .sparse import SparseGF2Matrix
from .core import rank, add, multiply
from .solvers import solve, nullspace


def pauli_group(n_qubits: int) -> List[Tuple[List[int], List[int]]]:
    """
    Generate elements of Pauli group on n qubits (Pauli operators).
    
    Returns:
        List of (x_part, z_part) tuples representing Pauli operators
    """
    pauli_ops = []

    # Generate all 4^n Pauli operators
    for i in range(4**n_qubits):
        x_part = []
        z_part = []

        val = i
        for qubit in range(n_qubits):
            pauli_type = val % 4
            val //= 4

            if pauli_type == 0:  # I
                x_part.append(0)
                z_part.append(0)
            elif pauli_type == 1:  # X
                x_part.append(1)
                z_part.append(0)
            elif pauli_type == 2:  # Y = iXZ
                x_part.append(1)
                z_part.append(1)
            elif pauli_type == 3:  # Z
                x_part.append(0)
                z_part.append(1)

        pauli_ops.append((x_part, z_part))

    return pauli_ops


def stabilizer_matrix(stabilizers: List[Tuple[List[int], List[int]]]) -> SparseGF2Matrix:
    """
    Create stabilizer matrix from list of Pauli stabilizers.
    
    Args:
        stabilizers: List of (x_part, z_part) tuples
        
    Returns:
        Stabilizer matrix in symplectic representation
    """
    if not stabilizers:
        return SparseGF2Matrix(0, 0)

    n_qubits = len(stabilizers[0][0])
    n_stabilizers = len(stabilizers)

    # Create matrix in symplectic form [X | Z]
    matrix_data = []
    for x_part, z_part in stabilizers:
        row = x_part + z_part  # Concatenate X and Z parts
        matrix_data.append(row)

    # Convert to sparse matrix
    coordinates = []
    for i, row in enumerate(matrix_data):
        for j, val in enumerate(row):
            if val == 1:
                coordinates.append((i, j))

    from .sparse import create_sparse_matrix
    return create_sparse_matrix(n_stabilizers, 2 * n_qubits, coordinates=coordinates)


def syndrome_decode(H: SparseGF2Matrix, syndrome: List[int]) -> Optional[List[int]]:
    """
    Decode error syndrome for quantum error correction.
    
    Args:
        H: Parity check matrix (X or Z stabilizers)
        syndrome: Measured syndrome vector
        
    Returns:
        Error pattern that produces the syndrome, or None if inconsistent
    """
    return solve(H, syndrome)


def commutator_gf2(pauli1: Tuple[List[int], List[int]], pauli2: Tuple[List[int], List[int]]) -> int:
    """
    Compute commutator of two Pauli operators over GF(2).
    
    Args:
        pauli1, pauli2: Pauli operators as (x_part, z_part)
        
    Returns:
        0 if operators commute, 1 if they anticommute
    """
    x1, z1 = pauli1
    x2, z2 = pauli2

    # Symplectic inner product: sum(x1 * z2 + z1 * x2) mod 2
    commutator = 0
    for i in range(len(x1)):
        commutator ^= (x1[i] & z2[i]) ^ (z1[i] & x2[i])

    return commutator


def stabilizer_rank(stabilizers: List[Tuple[List[int], List[int]]]) -> int:
    """
    Compute rank of stabilizer group.
    
    Args:
        stabilizers: List of stabilizer generators
        
    Returns:
        Number of independent stabilizers
    """
    if not stabilizers:
        return 0

    stab_matrix = stabilizer_matrix(stabilizers)
    return rank(stab_matrix)


def logical_operators(H_x: SparseGF2Matrix, H_z: SparseGF2Matrix) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Find logical X and Z operators for a quantum code.
    
    Args:
        H_x: X stabilizer matrix
        H_z: Z stabilizer matrix
        
    Returns:
        (logical_x_ops, logical_z_ops) as lists of operators
    """
    # Logical X operators are in nullspace of H_z
    logical_x = nullspace(H_z)

    # Logical Z operators are in nullspace of H_x
    logical_z = nullspace(H_x)

    return logical_x, logical_z


def code_distance_lower_bound(H_x: SparseGF2Matrix, H_z: SparseGF2Matrix) -> int:
    """
    Compute lower bound on quantum code distance.
    
    Args:
        H_x, H_z: X and Z stabilizer matrices
        
    Returns:
        Lower bound on code distance
    """
    # Find logical operators
    logical_x, logical_z = logical_operators(H_x, H_z)

    if not logical_x and not logical_z:
        return 0

    # Distance is minimum weight of logical operators
    min_weight = float('inf')

    for op in logical_x + logical_z:
        weight = sum(op)
        if weight > 0:
            min_weight = min(min_weight, weight)

    return int(min_weight) if min_weight != float('inf') else 0


def css_code_analysis(H1: SparseGF2Matrix, H2: SparseGF2Matrix) -> dict:
    """
    Analyze CSS code constructed from classical codes.
    
    Args:
        H1, H2: Classical parity check matrices
        
    Returns:
        Dictionary with code parameters
    """
    n1, n2 = H1.cols, H2.cols
    k1, k2 = n1 - rank(H1), n2 - rank(H2)

    # CSS code parameters
    n_physical = n1 + n2  # Total number of qubits
    n_logical = k1 + k2 - rank(H1) - rank(H2)  # Logical qubits (simplified)

    return {
        'n_physical': n_physical,
        'n_logical': max(0, n_logical),
        'rate': n_logical / n_physical if n_physical > 0 else 0,
        'classical_codes': {
            'H1': {
                'n': n1,
                'k': k1,
                'rank': rank(H1)
            },
            'H2': {
                'n': n2,
                'k': k2,
                'rank': rank(H2)
            }
        }
    }


def surface_code_analysis(distance: int) -> dict:
    """
    Analyze surface code parameters.
    
    Args:
        distance: Surface code distance
        
    Returns:
        Dictionary with code parameters
    """
    n_data = distance * distance
    n_x_checks = (distance - 1) * distance // 2
    n_z_checks = distance * (distance - 1) // 2
    n_logical = 1  # Surface code encodes 1 logical qubit

    return {
        'distance': distance,
        'n_physical': n_data,
        'n_logical': n_logical,
        'n_x_checks': n_x_checks,
        'n_z_checks': n_z_checks,
        'rate': n_logical / n_data,
        'threshold_estimate': 0.01  # Approximate error threshold
    }


def stabilizer_code_check(H_x: SparseGF2Matrix, H_z: SparseGF2Matrix) -> dict:
    """
    Check if matrices define a valid stabilizer code.
    
    Args:
        H_x, H_z: X and Z stabilizer matrices
        
    Returns:
        Dictionary with validation results
    """
    results = {'valid': True, 'errors': [], 'warnings': []}

    # Check dimensions
    if H_x.cols != H_z.cols:
        results['valid'] = False
        results['errors'].append("H_x and H_z must have same number of columns")

    n_qubits = H_x.cols

    # Check commutation relations: [H_x, H_z] = 0
    # This requires matrix multiplication over GF(2)
    try:
        # For CSS codes: H_x * H_z^T should be zero
        from .core import transpose
        H_z_T = transpose(H_z)
        commutator = multiply(H_x, H_z_T)

        # Check if result is zero matrix
        is_zero = True
        for i in range(commutator.rows):
            if commutator.get_row_bitwise(i) != 0:
                is_zero = False
                break

        if not is_zero:
            results['valid'] = False
            results['errors'].append("X and Z stabilizers do not commute")

    except Exception as e:
        results['warnings'].append(f"Could not check commutation: {e}")

    # Check rank conditions
    rank_x = rank(H_x)
    rank_z = rank(H_z)

    if rank_x + rank_z > n_qubits:
        results['warnings'].append("High rank sum may indicate overconstrained code")

    results['rank_x'] = rank_x
    results['rank_z'] = rank_z
    results['n_qubits'] = n_qubits

    return results
