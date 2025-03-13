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
        required_count = n - 1

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


# A matrix solver using the generic method (not bitwise)
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
    if isinstance(matrix, np.ndarray):
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


###############################################
# Main postprocessing and quantum circuit execution
###############################################

if __name__ == '__main__':
    # Define the secret string for Simon's algorithm (you can experiment with different strings)
    s = '10110100110010'
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

    # --- Simple selection of linearly independent vectors ---
    # Select n-1 linearly independent vectors from the sample list
    selected_vectors = []

    for vector in sample_list:
        # Skip if we already have enough vectors
        if len(selected_vectors) >= n - 1:
            break

        # Check if adding this vector keeps the set linearly independent
        test_set = selected_vectors + [vector]
        if is_independent_set(test_set):
            selected_vectors.append(vector)

    # Check if we found enough linearly independent vectors
    if len(selected_vectors) < n - 1:
        raise Exception(
            f"Could not find {n-1} linearly independent vectors. Rerun Simon's algorithm with more shots.")

    print("Selected linearly independent vectors ({} vectors):".format(len(selected_vectors)))
    print(selected_vectors)
    # Test if the selected vectors are linearly independent
    if not is_independent_set(selected_vectors):
        raise Exception("Error: The selected vectors are not linearly independent.")

    best_matrix = np.array(selected_vectors)

    # --- Compute the secret string using our bitwise GF(2) solver ---
    secret_str, elapsed_time = get_secret_integer_bitwise(best_matrix.tolist())
    print("Computed secret string (bitwise GF(2) solver):", secret_str)
    print("Expected secret string:", s)
    print("Time for nullspace computation: {:.9f} seconds".format(elapsed_time))

    # Add orthogonality check
    print("\n=== Orthogonality Check ===")
    expected_failures = 0
    computed_failures = 0

    # Convert secret strings to bit arrays
    expected_bits = [int(bit) for bit in s]
    computed_bits = [int(bit) for bit in secret_str]

    print("\nChecking vectors against expected secret:")
    for i, vector in enumerate(selected_vectors):
        dot_product = sum(expected_bits[j] & vector[j] for j in range(len(s))) % 2
        status = "✓" if dot_product == 0 else "✗"
        print(f"  Vector {i}: {status} (dot product = {dot_product})")
        if dot_product != 0:
            expected_failures += 1

    print(f"\nChecking vectors against computed secret:")
    for i, vector in enumerate(selected_vectors):
        dot_product = sum(computed_bits[j] & vector[j] for j in range(len(s))) % 2
        status = "✓" if dot_product == 0 else "✗"
        print(f"  Vector {i}: {status} (dot product = {dot_product})")
        if dot_product != 0:
            computed_failures += 1

    print(
        f"\nSummary: {expected_failures} failures with expected secret, {computed_failures} with computed secret"
    )

    if secret_str == s:
        print("Success: Found the correct secret string!")
    else:
        print("Error: The computed secret string is incorrect.")

    ###############################################
    # (Optional) Compare with the galois-based method
    ###############################################
    # GF = galois.GF(2)
    # # Convert best_matrix to a GF(2) matrix
    # gf_matrix = GF(best_matrix)
    # start_time = time.time()
    # null_space = gf_matrix.T.left_null_space()  # left null space using galois
    # galois_time = time.time() - start_time
    # null_vector = np.array(null_space)[0]
    # binary_string = "".join(null_vector.astype(int).astype(str))
    # print("Galois-based solver computed secret string:", binary_string)
    # print("Time for galois-based nullspace computation: {:.9f} seconds".format(galois_time))

    ###############################################
    # (Optional) Compare with the generic method
    ###############################################
    generic_str, generic_time = get_secret_integer_generic(best_matrix.tolist())
    print("Computed secret string (generic method):", generic_str)
    print("Time for generic nullspace computation: {:.9f} seconds".format(generic_time))

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

    ###  test the methods with smaller matrices
    # print("##########################")
    # m1 = [[1, 0, 1], [0, 1, 1], [1, 1, 0]]
    # print("Matrix 1:")
    # print(m1)
    # secret_str, elapsed_time = get_secret_integer_bitwise(m1)
    # print("Computed secret string (bitwise GF(2) solver):", secret_str)
    # print("Time for nullspace computation: {:.9f} seconds".format(elapsed_time))

    # secret_str, elapsed_time = get_secret_integer_generic(m1)
    # print("Computed secret string (generic method):", secret_str)
    # print("Time for generic nullspace computation: {:.9f} seconds".format(elapsed_time))

    # m2 = [[2, 3, 1, 5], [6, 1, 4, 1], [1, 2, 1, 3], [-1, 1, 1, 2]]
    # secret_str, elapsed_time = get_secret_integer_generic(m2)
    # print("Computed secret string (generic method):", secret_str)
    # print("Time for generic nullspace computation: {:.9f} seconds".format(elapsed_time))
