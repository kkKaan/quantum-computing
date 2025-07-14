"""
Optimized GF(2) Linear Algebra Functions
=======================================

This module contains the core bitwise optimizations for linear algebra over GF(2).
Extracted from Simon's algorithm postprocessing for general use.

Functions:
- pack_vector/unpack_vector: Convert between bit lists and integers
- gaussian_elimination_GF2_bitwise: Fast Gaussian elimination over GF(2)
- get_secret_integer_bitwise: Fast nullspace solver
- get_secret_integer_generic: Generic nullspace solver for comparison
"""

import time
from fractions import Fraction


def pack_vector(vec):
    """
    Pack a list of bits (e.g. [1, 0, 1, 1]) into an integer.
    The least-significant bit corresponds to the 0th element.
    """
    out = 0
    for i, bit in enumerate(vec):
        out |= (bit & 1) << i
    return out


def unpack_vector(x, n):
    """
    Unpack an integer x into a list of n bits.
    """
    return [(x >> i) & 1 for i in range(n)]


def gaussian_elimination_GF2_bitwise(rows, n):
    """
    Perform Gaussian elimination (row reduction) on a list of integers representing rows (each row is an n-bit number).
    Returns the row-echelon form and the list of pivot column indices.
    """
    # Make a copy
    A = rows[:]
    pivot_cols = []
    r = 0  # current row index in elimination
    for col in range(n):
        # Find pivot: look for a row from r onward that has a 1 in column col.
        pivot_row = None
        for i in range(r, len(A)):
            if (A[i] >> col) & 1:
                pivot_row = i
                break
        if pivot_row is None:
            continue  # no pivot in this column
        # Swap the pivot row into position r.
        A[r], A[pivot_row] = A[pivot_row], A[r]
        pivot_cols.append(col)
        # Eliminate the 1 in column col in all rows below.
        for i in range(r + 1, len(A)):
            if (A[i] >> col) & 1:
                A[i] ^= A[r]
        r += 1
        if r == len(A):
            break
    return A, pivot_cols


def nullspace_solution_bitwise(rows, pivot_cols, n):
    """
    Given the row-echelon form (as integers) and pivot columns,
    return a nontrivial solution vector x (as an integer) to A x = 0 mod 2.
    We force one free variable to 1.
    """
    all_cols = set(range(n))
    free_cols = sorted(list(all_cols - set(pivot_cols)))
    # If there is no free variable, then the only solution is the trivial one.
    # In Simon's algorithm we expect an underdetermined system so free_cols should not be empty.
    if not free_cols:
        raise ValueError("No free variable found; the system appears to be full rank.")
    # We'll choose the first free column (lowest index) and set it to 1.
    x = [0] * n
    x[free_cols[0]] = 1

    # Back-substitution in reverse order.
    # Our rows are in echelon form (not fully reduced), so we process pivot rows in reverse.
    # For each pivot row, let pivot be at col p. Then the row gives: x[p] + sum_{j in free cols, j > p with (row >> j) & 1} x[j] = 0.
    # We solve for x[p].
    num_pivots = len(pivot_cols)
    for i in reversed(range(num_pivots)):
        p = pivot_cols[i]
        sum_free = 0
        # Check bits in columns greater than p
        for j in range(p + 1, n):
            if (rows[i] >> j) & 1:
                sum_free ^= x[j]
        x[p] = sum_free  # since equation is: x[p] + sum = 0  => x[p] = sum (mod 2)
    # Convert x to integer representation.
    sol_int = pack_vector(x)
    # Ensure sol_int is nonzero.
    if sol_int == 0:
        # Force the free variable bit to 1 explicitly (should not happen normally).
        sol_int = 1 << free_cols[0]
    return sol_int


def get_secret_integer_bitwise(matrix):
    """
    Given a matrix (list of rows, each a list of bits) representing the independent
    measurement outcomes, pack them into integers, perform bitwise GF(2) elimination,
    and then solve for a nontrivial nullspace vector.
    Returns the solution as a binary string and the elapsed time.
    """
    n = len(matrix[0])
    # Pack rows into integers.
    rows = [pack_vector(row) for row in matrix]
    start_time = time.time()
    A_echelon, pivot_cols = gaussian_elimination_GF2_bitwise(rows, n)
    sol_int = nullspace_solution_bitwise(A_echelon, pivot_cols, n)
    elapsed_time = time.time() - start_time
    # Unpack the solution
    sol_bits = unpack_vector(sol_int, n)
    # Previously we reversed the bits; remove that reversal to use the order returned by unpack_vector.
    sol_str = "".join(str(b) for b in sol_bits)
    return sol_str, elapsed_time


def modinv(a, p):
    """
    Compute the modular inverse of a modulo p.
    Assumes p is prime.
    """
    # Extended Euclidean algorithm:
    t, new_t = 0, 1
    r, new_r = p, a % p
    while new_r != 0:
        quotient = r // new_r
        t, new_t = new_t, t - quotient * new_t
        r, new_r = new_r, r - quotient * new_r
    if r > 1:
        raise ValueError(f"{a} is not invertible modulo {p}")
    return t % p


def get_secret_integer_generic(matrix, mod=None):
    """
    A generic nullspace solver that uses our own Gaussian elimination routine.
    
    Parameters:
       matrix : a list of lists (or NumPy array) representing the matrix.
       mod    : if provided (e.g. mod=2), perform arithmetic modulo mod.
                If None and all entries are 0 or 1, then automatically use mod=2.
                
    Returns:
       A tuple (sol_str, elapsed_time) where sol_str is a string representation of one 
       nontrivial nullspace vector. If no free variable is found (i.e. full column rank), returns (None, elapsed_time).
    """
    start_time = time.time()

    # Convert input to list of lists
    if hasattr(matrix, 'tolist'):  # numpy array
        mat = matrix.tolist()
    else:
        mat = [list(row) for row in matrix]

    rows = len(mat)
    cols = len(mat[0]) if rows > 0 else 0

    # Automatically set mod=2 if mod is None and all entries are 0 or 1.
    if mod is None:
        is_binary = all((x in (0, 1) for row in mat for x in row))
        if is_binary:
            mod = 2

    # Convert matrix entries based on mod.
    if mod is None:
        # Exact arithmetic over Q
        A = [[Fraction(x) for x in row] for row in mat]
    else:
        # Work modulo mod (assumed prime)
        A = [[int(x) % mod for x in row] for row in mat]

    pivot_cols = []
    pivot_rows = []
    pivot_row = 0

    # Forward elimination (Gaussian elimination)
    for col in range(cols):
        pivot_found = False
        for r in range(pivot_row, rows):
            if mod is None:
                cond = (A[r][col] != 0)
            else:
                cond = (A[r][col] % mod != 0)
            if cond:
                pivot_found = True
                max_row = r
                break
        if not pivot_found:
            continue
        if max_row != pivot_row:
            A[pivot_row], A[max_row] = A[max_row], A[pivot_row]
        pivot_cols.append(col)
        pivot_rows.append(pivot_row)
        pivot_val = A[pivot_row][col]
        if mod is None:
            inv = Fraction(1) / pivot_val
        else:
            inv = modinv(pivot_val, mod)
        for c in range(col, cols):
            if mod is None:
                A[pivot_row][c] *= inv
            else:
                A[pivot_row][c] = (A[pivot_row][c] * inv) % mod
        for r in range(pivot_row + 1, rows):
            factor = A[r][col]
            if mod is None:
                if factor != 0:
                    for c in range(col, cols):
                        A[r][c] -= factor * A[pivot_row][c]
            else:
                if factor % mod != 0:
                    for c in range(col, cols):
                        A[r][c] = (A[r][c] - factor * A[pivot_row][c]) % mod
        pivot_row += 1
        if pivot_row == rows:
            break

    # Identify free columns
    all_cols = set(range(cols))
    free_cols = sorted(list(all_cols - set(pivot_cols)))
    if not free_cols:
        elapsed_time = time.time() - start_time
        return None, elapsed_time  # full column rank

    # Construct a nontrivial solution by setting one free variable to 1.
    if mod is None:
        solution = [Fraction(0) for _ in range(cols)]
        solution[free_cols[0]] = Fraction(1)
    else:
        solution = [0 for _ in range(cols)]
        solution[free_cols[0]] = 1

    # Back-substitution in reverse order.
    for i in reversed(range(len(pivot_cols))):
        p = pivot_cols[i]
        r_idx = pivot_rows[i]
        s_val = 0
        for j in range(p + 1, cols):
            if mod is None:
                s_val += A[r_idx][j] * solution[j]
            else:
                s_val = (s_val + A[r_idx][j] * solution[j]) % mod
        if mod is None:
            solution[p] = -s_val
        else:
            solution[p] = (-s_val) % mod

    # Normalize solution to a bit-string if possible.
    if mod is not None and mod == 2:
        sol_str = "".join("1" if solution[i] % 2 != 0 else "0" for i in range(cols))
    else:
        # Check if solution is binary (i.e. 0 or ±1 for exact arithmetic)
        is_binary = True
        for s_val in solution:
            if mod is None:
                if s_val != 0 and abs(s_val) != 1:
                    is_binary = False
                    break
            else:
                if s_val % mod not in (0, 1):
                    is_binary = False
                    break
        if is_binary:
            sol_str = "".join(
                "1" if (s_val != 0 if mod is None else s_val % mod != 0) else "0" for s_val in solution)
        else:
            sol_str = "(" + ", ".join(str(s_val) for s_val in solution) + ")"

    elapsed_time = time.time() - start_time
    return sol_str, elapsed_time


# Test function
def test_gf2_solver():
    """Quick test to verify the solver works correctly."""
    print("Testing GF(2) solver...")

    # Test matrix: should have nullspace vector [1, 1, 1]
    test_matrix = [[1, 1, 0], [1, 0, 1], [0, 1, 1]]

    # Test bitwise solver
    solution_bitwise, time_bitwise = get_secret_integer_bitwise(test_matrix)
    print(f"Bitwise solver: {solution_bitwise} (time: {time_bitwise:.6f}s)")

    # Test generic solver
    solution_generic, time_generic = get_secret_integer_generic(test_matrix)
    print(f"Generic solver: {solution_generic} (time: {time_generic:.6f}s)")

    print("Solver test completed!")


if __name__ == "__main__":
    test_gf2_solver()
