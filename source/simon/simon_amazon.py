from braket.circuits import Circuit, circuit
from braket.devices import LocalSimulator
from simons_utils import simons_oracle
import numpy as np
from sympy import Matrix

import matplotlib.pyplot as plt

# %matplotlib inline

# Sets the device to run the circuit on
device = LocalSimulator()

if __name__ == '__main__':
    s = '110001001'
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

    plt.bar(new_results.keys(), new_results.values())
    plt.xlabel('bit strings')
    plt.ylabel('counts')
    plt.xticks(rotation=70)
    # plt.show()

    # These new results are independent of the secret string s, and n linearly independent of them
    # can be used to solve for s

    # Classical post-processing to solve for s (from amazon)
    if len(new_results.keys()) < len(s):
        raise Exception('System will be underdetermined. Minimum ' + str(n) + ' bistrings needed, but only ' +
                        str(len(new_results.keys())) + ' returned. Please rerun Simon\'s algorithm.')
    string_list = []

    for key in new_results.keys():
        if key != "0" * n:
            string_list.append([int(c) for c in key])

    # string_list = [[1, 1, 1], [1, 1, 0], [0, 0, 1]]

    print('The result in matrix form is :')
    for a in string_list:
        print(a)  # to show all the bit strings

    M = Matrix(string_list).T

    ######

    # Construct the agumented matrix
    M_I = Matrix(np.hstack([M, np.eye(M.shape[0], dtype=int)]))

    # Perform row reduction, working modulo 2. We use the iszerofunc property of rref
    # to perform the Gaussian elimination over the finite field.
    M_I_rref = M_I.rref(iszerofunc=lambda x: x % 2 == 0)

    # In row reduced echelon form, we can end up with a solution outside of the finite field {0,1}.
    # Thus, we need to revert the matrix back to this field by treating fractions as a modular inverse.
    # Since the denominator will always be odd (i.e. 1 mod 2), it can be ignored.

    # Helper function to treat fractions as modular inverse:
    def mod2(x):
        return x.as_numer_denom()[0] % 2

    # Print MI_rref in a more readable format
    print('The row reduced echelon form of the augmented matrix is:')
    for row in M_I_rref[0].tolist():
        print(row)

    # Apply our helper function to the matrix
    M_I_final = M_I_rref[0].applyfunc(mod2)

    # Extract the kernel of M from the remaining columns of the last row, when s is nonzero.
    if all(value == 0 for value in M_I_final[-1, :M.shape[1]]):
        result_s = "".join(str(c) for c in M_I_final[-1, M.shape[1]:])

    # Otherwise, the sub-matrix will be full rank, so just set s=0...0
    else:
        result_s = '0' * M.shape[0]

    # Check whether result_s is equal to initial s:
    print('Secret string: ' + s)
    print('Result string: ' + result_s)
    if (result_s == s):
        print('We found the correct answer.')
    else:
        print('Error. The answer is wrong!')

    ######
