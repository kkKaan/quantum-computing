from braket.circuits import Circuit, circuit
from braket.devices import LocalSimulator
from simons_utils import simons_oracle
from fractions import Fraction
import numpy as np
import itertools
import time
import matplotlib.pyplot as plt

# Import the galois library if you want to compare timing (we use it only for comparison here)
import galois

# Sets the device to run the circuit on
device = LocalSimulator()

###############################################
# Helper functions for generating independent sets
###############################################


def get_all_combinations(vectors):
    """
    Generate all possible combinations (of size n) from the given list of vectors.
    """
    n = len(vectors[0])
    print('The number of vectors is: ' + str(len(vectors)))
    return list(itertools.combinations(vectors, n))


def is_independent_set(vectors):
    """
    Check whether a set of vectors (each as list of bits) is linearly independent
    over the reals (which agrees with GF(2) independence for 0/1 vectors).
    """
    matrix = np.array(vectors, dtype=float)
    rank = np.linalg.matrix_rank(matrix)
    return (rank == len(vectors))


def get_independent_set(samples):
    """
    From a list of measurement bitstrings (each as a list of ints), return the first combination 
    of size n that is linearly independent.
    """
    all_combinations = get_all_combinations(samples)
    for combination in all_combinations:
        if is_independent_set(combination):
            return combination
    return None


def get_all_independent_sets(samples):
    """
    Return all combinations (of size n) that form an independent set.
    """
    all_combinations = get_all_combinations(samples)
    independent_sets = []
    for combination in all_combinations:
        if is_independent_set(combination):
            independent_sets.append(combination)
    print('The number of independent sets is: ' + str(len(independent_sets)))
    return independent_sets


def calculate_density(matrix):
    """
    Calculate the density (number of nonzero entries) of a given 2D numpy array.
    """
    return np.count_nonzero(matrix)


def find_most_sparse_matrix(matrices):
    """
    From a list of matrices (as numpy arrays), return the one with the smallest density.
    """
    min_density = np.inf
    most_sparse_matrix = None
    for matrix in matrices:
        density = np.count_nonzero(matrix)
        if density < min_density:
            min_density = density
            most_sparse_matrix = matrix
    return most_sparse_matrix


import random


def find_sparse_enough_matrix_greedy(samples, threshold, required_count=None, max_trials=1000):
    """
    Given a list of sample vectors (each sample is a list of bits),
    try to greedily select a subset (of size 'required_count') that is linearly independent
    (using is_independent_set) and whose density (total number of ones in the candidate matrix)
    is below the threshold.
    
    Parameters:
      samples: list of lists, each inner list is a bit string (list of ints)
      threshold: density threshold (number of ones in the candidate matrix)
      required_count: number of rows required (default is len(samples[0]), i.e. bit-length)
      max_trials: maximum number of random order trials to attempt
      
    Returns:
      A candidate independent set (as a list of bit-string lists) if one with density <= threshold
      is found; otherwise, returns the candidate with the lowest density found.
    """
    if not samples:
        return None

    n = len(samples[0])
    if required_count is None:
        required_count = n  # by default, we want n independent rows

    best_candidate = None
    best_density = float('inf')

    # Try multiple random orderings.
    for _ in range(max_trials):
        # Make a copy and shuffle randomly.
        candidate_order = samples.copy()
        random.shuffle(candidate_order)

        candidate = []
        # Greedily add a vector if it keeps the set independent.
        for vec in candidate_order:
            if not candidate:
                candidate.append(vec)
            else:
                # Check if candidate + current vector is independent.
                if is_independent_set(candidate + [vec]):
                    candidate.append(vec)
            if len(candidate) == required_count:
                break

        # Only consider candidate if it reached required_count.
        if len(candidate) == required_count:
            density = calculate_density(np.array(candidate))
            if density <= threshold:
                return candidate
            if density < best_density:
                best_density = density
                best_candidate = candidate[:]

    return best_candidate


###############################################
# Bitwise Gaussian Elimination routines over GF(2)
###############################################


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


def convert_to_fraction_matrix(A):
    """
    Convert a NumPy array A into an m×n array of Fractions.
    """
    m, n = A.shape
    B = np.empty((m, n), dtype=object)
    for i in range(m):
        for j in range(n):
            # Convert each entry to Fraction. (If already Fraction, this is idempotent.)
            B[i, j] = Fraction(A[i, j])
    return B


def rref_generic_float(A, tol=1e-12):
    """
    Compute the RREF of A (a NumPy array) using float arithmetic.
    Returns the RREF and the list of pivot column indices.
    """
    R = A.copy().astype(np.float64)
    m, n = R.shape
    pivot_cols = []
    r = 0
    for col in range(n):
        # Find a pivot in column col (from row r downward)
        pivot_row = None
        max_val = tol
        for i in range(r, m):
            if abs(R[i, col]) > max_val:
                max_val = abs(R[i, col])
                pivot_row = i
        if pivot_row is None:
            continue  # no pivot in this column
        # Swap pivot row into position r
        if pivot_row != r:
            R[[r, pivot_row]] = R[[pivot_row, r]]
        pivot_cols.append(col)
        # Normalize pivot row
        R[r, :] = R[r, :] / R[r, col]
        # Eliminate column col in all other rows
        for i in range(m):
            if i != r and abs(R[i, col]) > tol:
                R[i, :] = R[i, :] - R[i, col] * R[r, :]
        r += 1
        if r == m:
            break
    return R, pivot_cols


def rref_generic_exact(B):
    """
    Compute the RREF of matrix B using exact arithmetic with Fractions.
    B is assumed to be a NumPy array of object (Fractions).
    Returns (R, pivot_cols) where R is the RREF (as an m×n array of Fractions).
    """
    B = B.copy()  # make a copy (entries are Fractions)
    m, n = B.shape
    pivot_cols = []
    r = 0
    for col in range(n):
        pivot_row = None
        for i in range(r, m):
            if B[i, col] != 0:
                pivot_row = i
                break
        if pivot_row is None:
            continue
        if pivot_row != r:
            B[[r, pivot_row]] = B[[pivot_row, r]]
        pivot_cols.append(col)
        pivot_val = B[r, col]
        # Normalize pivot row: divide the entire row by pivot_val
        for j in range(col, n):
            B[r, j] = B[r, j] / pivot_val
        # Eliminate this column from all other rows
        for i in range(m):
            if i != r and B[i, col] != 0:
                factor = B[i, col]
                for j in range(col, n):
                    B[i, j] = B[i, j] - factor * B[r, j]
        r += 1
        if r == m:
            break
    return B, pivot_cols


def solve_nullspace_generic(A, use_exact=False, tol=1e-12):
    """
    Compute a basis for the nullspace (kernel) of the matrix A (an m×n NumPy array)
    without using any external nullspace library.
    
    Parameters:
      A         : 2D NumPy array (of numbers). For exact (symbolic) arithmetic, the entries
                  should be integers or rationals.
      use_exact : If True, the algorithm converts the matrix to one with Fraction entries and
                  works exactly. Otherwise, standard float arithmetic is used.
      tol       : Tolerance for determining zero in float arithmetic (ignored in exact mode).
    
    Returns:
      A list of basis vectors. Each basis vector is represented as a 1D NumPy array.
      (If the nullspace is trivial, returns an empty list.)
    """
    m, n = A.shape
    if use_exact:
        B = convert_to_fraction_matrix(A)
        R, pivot_cols = rref_generic_exact(B)
    else:
        R, pivot_cols = rref_generic_float(A, tol)

    free_cols = [j for j in range(n) if j not in pivot_cols]
    if len(free_cols) == 0:
        return []  # full column rank

    basis = []
    # For each free variable, we set that variable to 1 and all other free variables to 0,
    # then solve for the pivot variables via back substitution.
    for free in free_cols:
        # Initialize a solution vector x of length n (using Fraction(0) in exact mode)
        if use_exact:
            x = [Fraction(0) for _ in range(n)]
            one = Fraction(1)
        else:
            x = [0.0 for _ in range(n)]
            one = 1.0
        x[free] = one
        # For each pivot row r_idx corresponding to pivot column p:
        # The equation from row r_idx is:
        #   x[p] + sum_{j in free_cols, j > p} R[r_idx, j]*x[j] = 0
        # so we solve for x[p] = - sum_{j in free_cols} R[r_idx, j]*x[j]
        for r_idx, p in enumerate(pivot_cols):
            # Only process if the pivot column p is less than free;
            # note: since our RREF is computed columnwise, the pivot rows are in order.
            # (You may also compute in reverse order.)
            s = one * 0  # initialize sum to 0 (works for both Fraction and float)
            for j in free_cols:
                # In exact mode, use Fraction arithmetic; in float mode, float multiplication.
                s += R[r_idx, j] * x[j]
            x[p] = -s
        # Append x as a NumPy array. If in exact mode, you may wish to convert the Fractions
        # to floats or strings for display.
        basis.append(np.array(x, dtype=object if use_exact else float))
    return basis


###############################################
# Main postprocessing and quantum circuit execution
###############################################

if __name__ == '__main__':
    # Define the secret string for Simon's algorithm (you can experiment with different strings)
    s = '1001110010'
    n = len(s)

    circ = Circuit()
    # Apply Hadamard gates to first n qubits
    circ.h(range(n))
    # Apply the oracle (this function is assumed to be implemented in simons_utils)
    circ.simons_oracle(s)
    # Apply Hadamard gates to the first n qubits
    circ.h(range(n))
    # Uncomment to see the circuit:
    # print(circ)

    # Run the circuit with a shot count that is 4*n (or higher if needed)
    task = device.run(circ, shots=4 * n)
    result = task.result()
    counts = result.measurement_counts

    # Process the measurement outcomes: keep only the first n qubits per outcome.
    new_results = {}
    for bitstring, count in counts.items():
        trunc_bitstring = bitstring[:n]
        new_results[trunc_bitstring] = new_results.get(trunc_bitstring, 0) + count

    # (Optional) Plot a histogram of the measurement counts.
    # plt.bar(new_results.keys(), new_results.values())
    # plt.xlabel('bit strings (first n qubits)')
    # plt.ylabel('counts')
    # plt.xticks(rotation=70)
    # plt.show()

    # Extract the measurement bitstrings (as lists of ints), ignoring the all-zeros result.
    sample_list = []
    for key in new_results.keys():
        if key != "0" * n:
            sample_list.append([int(c) for c in key])

    # Check that we have at least n measurements
    if len(sample_list) < n:
        raise Exception(
            f"System underdetermined: need at least {n} distinct bitstrings, but got {len(sample_list)}. Rerun Simon's algorithm."
        )

    # --- Redundancy and sparsity selection ---
    # # Get all independent sets of size n from the samples.
    # independent_sets = get_all_independent_sets(sample_list)
    # if not independent_sets:
    #     raise Exception(
    #         "No independent set found; please increase the shot count or rerun Simon's algorithm.")

    # # Convert each independent set (a tuple of n bitstrings) into a numpy array (each row is a vector)
    # matrices = [np.array(ind_set) for ind_set in independent_sets]

    # Choose the one with the lowest density (fewest ones) – a heuristic that might make elimination faster.
    best_matrix = find_sparse_enough_matrix_greedy(
        sample_list, 3 * len(sample_list[0]))  # find_most_sparse_matrix(matrices) # -> only when s is small
    print("Chosen independent set (matrix form):")
    print(best_matrix)

    best_matrix = np.array(best_matrix)

    # --- Compute the secret string using our bitwise GF(2) solver ---
    secret_str, elapsed_time = get_secret_integer_bitwise(best_matrix.tolist())
    print("Computed secret string (bitwise GF(2) solver):", secret_str)
    print("Expected secret string:", s)
    print("Time for nullspace computation: {:.9f} seconds".format(elapsed_time))

    if secret_str == s:
        print("Success: Found the correct secret string!")
    else:
        print("Error: The computed secret string is incorrect.")

    ###############################################
    # (Optional) Compare with the galois-based method
    ###############################################
    GF = galois.GF(2)
    # Convert best_matrix to a GF(2) matrix
    gf_matrix = GF(best_matrix)
    start_time = time.time()
    null_space = gf_matrix.T.left_null_space()  # left null space using galois
    galois_time = time.time() - start_time
    null_vector = np.array(null_space)[0]
    binary_string = "".join(null_vector.astype(int).astype(str))
    print("Galois-based solver computed secret string:", binary_string)
    print("Time for galois-based nullspace computation: {:.9f} seconds".format(galois_time))

    ###############################################
    # (Optional) Compare with the generic method
    ###############################################
    # Convert best_matrix to a NumPy array of floats
    float_matrix = best_matrix.astype(float)
    start_time = time.time()
    nullspace_basis = solve_nullspace_generic(float_matrix)
    generic_time = time.time() - start_time
    if nullspace_basis:
        null_vector = nullspace_basis[0]
        binary_string = "".join(null_vector.astype(int).astype(str))
        print("Generic solver computed secret string:", binary_string)
    else:
        print("Generic solver: No nontrivial nullspace found.")

    ###############################################
    # Compare different sparsities, for all possible matrices, with bitwise solver
    ###############################################
    # densities = []
    # times = []
    # for matrix in matrices:
    #     start_time = time.time()
    #     secret_str, elapsed_time = get_secret_integer_bitwise(matrix.tolist())
    #     densities.append(calculate_density(matrix))
    #     times.append(elapsed_time)

    # # Plot the density vs. time
    # plt.scatter(densities, times)
    # plt.xlabel("Density (number of ones)")
    # plt.ylabel("Time (s)")
    # plt.title("Density vs. time for bitwise GF(2) solver")
    # plt.show()
