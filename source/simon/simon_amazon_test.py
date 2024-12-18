from braket.circuits import Circuit, circuit
from braket.devices import LocalSimulator
from simons_utils import simons_oracle
import numpy as np
from sympy import Matrix
import galois
import itertools
import time

import matplotlib.pyplot as plt

# %matplotlib inline

# Sets the device to run the circuit on
device = LocalSimulator()


def get_all_combinations(vectors):
    """
    The following function generates all possible combinations of the vectors
    """
    n = len(vectors[0])
    print('The number of vectors is: ' + str(len(vectors)))
    return list(itertools.combinations(vectors, n))


def is_independent_set(vectors):
    """
    The following function checks whether a set contains linearly independet vectors
    """
    matrix = np.array(vectors, dtype=float)  # Ensure numerical data type
    rank = np.linalg.matrix_rank(matrix)
    # print('The rank of the matrix is: ' + str(rank))
    if rank == len(vectors):
        return True
    else:
        return False


def get_independent_set(samples):
    """
    The following function gets samples of n-sized strings from running the quantum part and return an n-1 x n matrix,
    whose rows forms a set if independent
    """
    all_combinations = get_all_combinations(samples)
    for combination in all_combinations:
        if is_independent_set(combination):
            return combination


def get_all_independent_sets(samples):
    """
    The following function gets samples of n-sized strings from running the quantum part and return an n-1 x n matrix,
    """
    all_combinations = get_all_combinations(samples)
    independent_sets = []
    for combination in all_combinations:
        if is_independent_set(combination):
            independent_sets.append(combination)
    return independent_sets


def calculate_density(matrix):
    """
    The following function calculates the density of a matrix
    """
    return np.count_nonzero(matrix)


def find_most_sparse_matrix(matrices):
    """
    The following function finds the most sparse matrix from a set of matrices

    Parameters:
        matrices: list of "numpy" arrays

    Returns:
        numpy array: the most sparse matrix
    """
    min_density = 1
    most_sparse_matrix = None
    for matrix in matrices:
        density = np.count_nonzero(matrix)
        if density < min_density:
            min_density = density
            most_sparse_matrix = matrix
    return most_sparse_matrix


def get_secret_integer(matrix):
    """
    Computes the secret integer from the left null space of a matrix over GF(2).
    """
    gf_v = GF(matrix)  # Converting to a matrix over Z_2

    # print('The matrix over GF(2) is:')
    # for a in gf_v:
    #     print(a)

    ### Keep the time of the computation below
    start_time = time.time()
    null_space = gf_v.T.left_null_space()  # Finding the left null space of the matrix
    total_time = time.time() - start_time
    # print("--- %s seconds ---" % total_time)  # Print the time taken for the computation
    ###

    null_vector = np.array(null_space)[0]  # Extract the first null space vector
    binary_string = "".join(null_vector.astype(int).astype(str))  # Ensure binary conversion
    return binary_string, total_time


if __name__ == '__main__':
    s = '10010110'
    # Other examples to try:
    # s = '011'
    # s = '00000'
    # s = '1'
    # Generate a random string of random length from 1 to 10:
    # s="".join(str(np.random.randint(2)) for _ in range(np.random.randint(1,10)))
    # print("The secret string is: " + s)

    n = len(s)
    circ = Circuit()

    # Apply Hadamard gates to first n qubits
    circ.h(range(n))

    # Now apply the Oracle for f
    circ.simons_oracle(s)

    # Apply Hadamard gates to the first n qubits
    circ.h(range(n))
    # print(circ)

    task = device.run(circ, shots=4 * n)
    result = task.result()  # results of all qubits (2n), we need to use only the first n qubits
    counts = result.measurement_counts
    # plt.bar(counts.keys(), counts.values())
    # plt.xlabel('bit strings')
    # plt.ylabel('counts')
    # plt.xticks(rotation=90)
    # plt.show()

    new_results = {}
    for bitstring, count in result.measurement_counts.items():
        # Only keep the outcomes on first n qubits
        trunc_bitstring = bitstring[:n]
        # Add the count to that of the of truncated bit string
        new_results[trunc_bitstring] = new_results.get(trunc_bitstring, 0) + count

    # plt.bar(new_results.keys(), new_results.values())
    # plt.xlabel('bit strings')
    # plt.ylabel('counts')
    # plt.xticks(rotation=70)
    # plt.show()

    # These new results are independent of the secret string s, and n linearly independent of them
    # can be used to solve for s

    ##### Classical post-processing to solve for s
    if len(new_results.keys()) < len(s):
        raise Exception('System will be underdetermined. Minimum ' + str(n) + ' bistrings needed, but only ' +
                        str(len(new_results.keys())) + ' returned. Please rerun Simon\'s algorithm.')
    string_list = []

    for key in new_results.keys():
        if key != "0" * n:
            string_list.append([int(c) for c in key])

    # print('The result in matrix form is :')
    # for a in string_list:
    #     print(a)  # to show all the bit strings

    GF = galois.GF(2)

    # Get the independent set
    ind_v = get_independent_set(string_list)
    print('The independent set is:')
    for a in ind_v:
        print(a)

    # Get the secret integer
    secret_int, passed_time = get_secret_integer(ind_v)
    print('Found secret string is: ' + str(secret_int))
    print('The secret string is: ' + s)
    if (secret_int == s):
        print('We found the correct answer.')
    else:
        print('Error. The answer is wrong!')

    # print("test gf")

    # Test the Galois Field
    # gf = galois.GF(11)

    # # create 6x6 matrix with numbers smaller than 11
    # test_matrix = np.random.randint(11, size=(6, 6))
    # print('The matrix is:')
    # for a in test_matrix:
    #     print(a)

    # gf_v = gf(test_matrix)
    # print('The matrix over GF(11) is:')
    # for a in gf_v:
    #     print(a)

    # start_time = time.time()
    # null_space = gf_v.T.left_null_space()
    # print("--- %s seconds ---" % (time.time() - start_time))
    # print('The left null space of the matrix is:')
    # for a in null_space:
    #     print(a)

    print("------------------------------")

    independent_sets = get_all_independent_sets(string_list)
    print('The number of independent sets is: ' + str(len(independent_sets)))
    times_and_density = []
    for matrix in independent_sets:
        # Get sparcity of the matrix and total time passed, give a graph showing these
        density = calculate_density(matrix)
        found_secret, found_time = get_secret_integer(matrix)
        if found_secret == s:
            times_and_density.append((density, found_time))

    # Sort the list of tuples based on the density
    times_and_density.sort(key=lambda x: x[0])

    # Plot the graph
    x = [x[0] for x in times_and_density]
    y = [x[1] for x in times_and_density]
    plt.plot(x, y)
    plt.xlabel('Density')
    plt.ylabel('Time')
    plt.yscale('log')
    plt.show()
