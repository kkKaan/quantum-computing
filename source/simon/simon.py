from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram, plot_distribution
import matplotlib.pyplot as plt
from qiskit.quantum_info import Operator
import numpy as np
from sympy import Matrix, GF
import sympy
import itertools
from collections import Counter
import scipy as sp
import datetime
import galois

from qiskit.circuit.library import GroverOperator, MCMT, ZGate

# Imports from Qiskit Runtime
from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit.primitives import Sampler


class Batch(Session):
    """Class for creating a batch mode in Qiskit Runtime."""

    pass


def simon_oracle(b):
    """
    Returns a Simon oracle for bitstring b

    :param b: The bitstring for the Simon oracle
    :return: The Simon oracle as a QuantumCircuit
    """
    b = b[::-1]  # reverse b for easy iteration
    n = len(b)
    qc = QuantumCircuit(n * 4)
    # Do copy; |x>|0> -> |x>|x>
    for q in range(n):
        qc.cx(q, q + n)
    if '1' not in b:
        return qc  # 1:1 mapping, so just exit
    i = b.find('1')  # index of first non-zero bit in b
    # Do |x> -> |s.x> on condition that q_i is 1
    for q in range(n):
        if b[q] == '1':
            qc.cx(i, (q) + n)
    return qc


def simon_circuit(s):
    """
    Creates and executes Simon's circuit for a given secret string s.

    :param s: The secret string for the Simon oracle
    :return: The final simon circuit with the secret string s
    """
    n = len(s)
    simon_circuit = QuantumCircuit(n * 4, n)

    simon_circuit.h(range(n))

    simon_circuit.barrier()
    simon_circuit &= simon_oracle(s)  # Adding the oracle to the circuit
    simon_circuit.barrier()

    simon_circuit.h(range(n))
    simon_circuit.measure(range(n), range(n))
    # print(simon_circuit)
    return simon_circuit


def run_simon_circuit(simon_circuit, shots):
    """
    Executes the Simon circuit on a quantum simulator.

    :param simon_circuit: The Simon circuit to run
    :param shots: The number of times to run the simulation
    :return: The result of the simulation as a histogram of bitstrings
    """
    simulator = Aer.get_backend('qasm_simulator')
    compiled_circuit = transpile(simon_circuit, simulator)
    job = simulator.run(compiled_circuit, shots=shots)
    result = job.result()
    counts = result.get_counts()
    counts = [key for key in counts.keys() if key != '0' * len(key)]
    return counts


def is_fully_independent(matrix):
    """
    Check if the matrix is of full rank.

    :param matrix: A numpy array representing the matrix.
    :return: True if the matrix is of full rank, False otherwise.
    """
    if matrix.shape[0] != matrix.shape[1]:
        return False
    return np.linalg.matrix_rank(matrix) == matrix.shape[0]


def find_all_independent_subsets(equations, n):
    """
    Finds all subsets of size n that are linearly independent.

    :param equations: List of equations, where each is represented as a bit string.
    :param n: Size of each subset, and the number of bits in each equation.
    :return: A list of all independent subsets, each subset is a list of bit strings.
    """
    all_subsets = list(itertools.combinations(equations, n))
    print("Number of subsets:")
    print(len(all_subsets))
    print(all_subsets[:3])
    a = 0
    independent_subsets = []
    for subset in all_subsets:
        # Convert subset to matrix
        matrix = np.array([[int(bit) for bit in eq] for eq in subset])
        # if a == 0:
        #     print(matrix)
        #     a += 1
        if is_fully_independent(matrix):
            independent_subsets.append(subset)
    return independent_subsets


def adotz(a, z):
    """
    Computes the dot product of two binary strings a and z.

    :param a: The first binary string
    :param z: The second binary string
    :return: The dot product of a and z
    """
    return sum([int(a[i]) * int(z[i]) for i in range(len(a))]) % 2


if __name__ == "__main__":
    s = '11010'
    n = len(s)
    equations = []
    circuit = simon_circuit(s)
    A = np.empty((0, len(s)), int)

    equations = run_simon_circuit(circuit, 10000)
    print("Equations:")
    print(equations)

    # Check each element of the equations with the others with the adotz function
    for i in range(len(equations)):
        results = []
        for j in range(len(equations)):
            if i != j:
                results.append(adotz(equations[i], equations[j]))
                # print(f"{equations[i]}.{equations[j]} = ", adotz(equations[i], equations[j]))

        if sum(results) == 0:
            print(f"Secret string found: {equations[i]}")
            break

    # for elem in equations:
    #     print(f"{s}.{elem} = ", adotz(s, elem))

    # Find all subsets of size n
    all_subsets = list(itertools.combinations(equations, n))

    # # Take the time for adotz function with each subset
    # times = {}
    # for subset in all_subsets:
    #     subset_results = []
    #     start = datetime.datetime.now().timestamp()
    #     for elem in subset:
    #         # Start initial time
    #         subset_results.append(adotz(s, elem))
    #         # End time

    #     end = datetime.datetime.now().timestamp()
    #     if sum(subset_results) == 0:
    #         times[subset] = end - start

    #     print("Number of subsets:")
    #     print(len(all_subsets))
    #     print("Subsets")
    #     print(subset_results)

    # print("Times:")
    # print(times)

    # # Find all independent subsets of size n
    # independent_subsets = find_all_independent_subsets(equations, n)
    # print("Independent Subsets:")
    # print(len(independent_subsets))

    # test linear independence
    # A = np.array([[1, 0, 1], [1, 1, 1], [0, 1, 0]])
    # print(is_fully_independent(A))

    # resultss = []
    # secret = np.random.randint(2, size=n)
    # secret_str = ''.join([str(bit) for bit in secret])

    # n_samples = 100
    # for _ in range(n_samples):
    #     flag = False
    #     while not flag:
    #         results = run_simon_circuit(circuit, 2 * n)
    #         flag = post_processing(results, resultss)

    # freqs = Counter(resultss)
    # print(f"Most common results: {freqs.most_common(1)[0]}")

    # To run on hardware, select the backend with the fewest number of jobs in the queue
    # service = QiskitRuntimeService(channel="ibm_quantum", token="dac892343da53c40e1fea5dbe253c50570450f29e05045767a44f761cf49ad52ab1f95955043e2c6b6e4116a636c8e6fd3ef5edc5e443e45ac77df4fc17a7880")
    # backend = service.least_busy(operational=True, simulator=False)

    # with Batch(backend=backend) as batch:
    #     sampler = Sampler()
    #     dist = sampler.run(circuit, shots=10000).result().quasi_dists[0]

    # print(dist)
    # fig = plot_distribution(dist)
    # plt.savefig("simon_distribution.png")

    # while len(equations) < len(s):
    #     current_eqn = run_simon_circuit(simon_circuit(s), 1)
    #     # print(current_eqn)
    #     if len(current_eqn) == 0:
    #         continue
    #     # Check linear independence and other conditions
    #     if current_eqn[0] not in equations and current_eqn[0] != '0' * len(s) and is_independent(current_eqn[0], A):
    #         equations.extend(current_eqn)
    #         print(equations)
    #         A = np.vstack([A, np.array([int(bit) for bit in current_eqn[0]])])

    # Solve the system of equations
    # sym_A = Matrix(A)
    # b = sym_A.solve(Matrix([0] * len(s)))
    # print("Secret string:")
    # print(b)

    # ns = nullspace(A)
    # print(ns)

    # A = np.empty((0, len(s)), int)
    # for bitstring in equations:
    #     new_equation = np.array([int(bit) for bit in bitstring])
    #     if is_independent(new_equation, A):
    #         A = np.vstack([A, new_equation])

    # print("Equations:")
    # print(equations)
    # print("Independent Equations Matrix:")
    # print(A)

    # A_sympy = Matrix(A)
    # nullspace = _nullspace(A_sympy)
    # print("Nullspace:")
    # print(nullspace)
