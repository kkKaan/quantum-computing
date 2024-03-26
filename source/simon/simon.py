from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram, plot_distribution
import matplotlib.pyplot as plt
from qiskit.quantum_info import Operator
import numpy as np
from sympy import Matrix, GF

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
    b = b[::-1] # reverse b for easy iteration
    n = len(b)
    qc = QuantumCircuit(n * 2)
    # Do copy; |x>|0> -> |x>|x>
    for q in range(n):
        qc.cx(q, q+n)
    if '1' not in b: 
        return qc  # 1:1 mapping, so just exit
    i = b.find('1') # index of first non-zero bit in b
    # Do |x> -> |s.x> on condition that q_i is 1
    for q in range(n):
        if b[q] == '1':
            qc.cx(i, (q)+n)
    return qc 

def simon_circuit(s):
    """
    Creates and executes Simon's circuit for a given secret string s.

    :param s: The secret string for the Simon oracle
    :return: The final simon circuit with the secret string s
    """
    n = len(s)
    simon_circuit = QuantumCircuit(n * 2, n)

    simon_circuit.h(range(n))    

    simon_circuit.barrier()
    simon_circuit &= simon_oracle(s) # Adding the oracle to the circuit
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

def is_independent(new_equation, matrix):
    """
    Checks if a new equation is independent of a given matrix.

    :param new_equation: The new equation to check
    :param matrix: The matrix to check against
    :return: True if the new equation is independent, False otherwise
    """
    if matrix.size == 0:
        return True
    # Convert the bit string to an array of integers
    new_row = np.array([int(bit) for bit in new_equation]).reshape(1, -1)
    augmented_matrix = np.vstack([matrix, new_row])
    rank_before = np.linalg.matrix_rank(matrix, tol=None)
    rank_after = np.linalg.matrix_rank(augmented_matrix, tol=None)
    return rank_after > rank_before

# def _iszero(x):
#     """
#     Returns True if x is zero.
#     """
#     return getattr(x, 'is_zero', None)

# def _nullspace(M, simplify=False, iszerofunc=_iszero):
#     """
#     Returns list of vectors (Matrix objects) that span nullspace of ``M``

#     Examples
#     ========

#     >>> from sympy import Matrix
#     >>> M = Matrix(3, 3, [1, 3, 0, -2, -6, 0, 3, 9, 6])
#     >>> M
#     Matrix([
#     [ 1,  3, 0],
#     [-2, -6, 0],
#     [ 3,  9, 6]])
#     >>> M.nullspace()
#     [Matrix([
#     [-3],
#     [ 1],
#     [ 0]])]

#     See Also
#     ========

#     columnspace
#     rowspace
#     """

#     reduced, pivots = M.rref(iszerofunc=iszerofunc, simplify=simplify)

#     free_vars = [i for i in range(M.cols) if i not in pivots]
#     basis     = []

#     for free_var in free_vars:
#         # for each free variable, we will set it to 1 and all others
#         # to 0.  Then, we will use back substitution to solve the system
#         vec           = [M.zero] * M.cols
#         vec[free_var] = M.one

#         for piv_row, piv_col in enumerate(pivots):
#             vec[piv_col] -= reduced[piv_row, free_var]

#         basis.append(vec)

#     return [M._new(M.cols, 1, b) for b in basis]

# def rref(A, tol=None):
#     """
#     Compute the Reduced Row Echelon Form of matrix A.

#     Parameters:
#     A (np.array): The input matrix.
#     tol (float, optional): Tolerance to consider an element in A as zero.

#     Returns:
#     np.array: The RREF form of A.
#     """
#     # Convert to float type to prevent integer division
#     A = A.astype(np.float64)
#     rows, cols = A.shape
#     r = 0  # Rank of A
#     pivots_pos = []  # Positions of pivot elements
#     for c in range(cols):
#         # Find the pivot row
#         pivot = np.argmax(np.abs(A[r:rows, c])) + r
#         m = np.abs(A[pivot, c])
#         if m <= tol:
#             # Skip column c, making it zero below pivot
#             A[r:rows, c] = 0
#             continue

#         # Swap current row and pivot row
#         A[[r, pivot], c:cols] = A[[pivot, r], c:cols]

#         # Normalize pivot row
#         A[r, c:cols] = A[r, c:cols] / A[r, c]

#         # Eliminate below
#         v = A[r, c:cols]  # Copy pivot row
#         if r < rows - 1:  # If not the last row
#             A[r + 1:rows, c:cols] -= v * A[r + 1:rows, c:c + 1]

#         # Eliminate above
#         if r > 0:
#             A[0:r, c:cols] -= v * A[0:r, c:c + 1]

#         pivots_pos.append(r)
#         r += 1
#         if r == rows:
#             break

#     return A, pivots_pos

# def nullspace(A, tol=1e-12):
#     """
#     Find the nullspace of A based on RREF form.

#     Parameters:
#     A (np.array): The input matrix.
#     tol (float): Tolerance for considering an element zero.

#     Returns:
#     np.array: Basis for the nullspace of A.
#     """
#     rref_matrix, pivots_pos = rref(A, tol)
#     rows, cols = A.shape
#     r = len(pivots_pos)
#     free_vars = [j for j in range(cols) if j not in pivots_pos]

#     # Initialize nullspace matrix
#     null_space = np.zeros((cols, len(free_vars)), dtype=np.float64)

#     # Set free variables to 1 one at a time and solve for the pivot variables
#     for i, free_var in enumerate(free_vars):
#         null_space[free_var, i] = 1
#         for pivot, row in zip(pivots_pos[::-1], range(r - 1, -1, -1)):
#             null_space[pivot, i] = -np.dot(rref_matrix[row, pivot + 1:], null_space[pivot + 1:, i])

#     return null_space

if __name__ == "__main__":
    s = '1001101000'
    equations = []
    circuit = simon_circuit(s)
    A = np.empty((0, len(s)), int)

    # To run on hardware, select the backend with the fewest number of jobs in the queue
    # service = QiskitRuntimeService(channel="ibm_quantum", token="dac892343da53c40e1fea5dbe253c50570450f29e05045767a44f761cf49ad52ab1f95955043e2c6b6e4116a636c8e6fd3ef5edc5e443e45ac77df4fc17a7880")
    # backend = service.least_busy(operational=True, simulator=False)

    # with Batch(backend=backend) as batch:
    #     sampler = Sampler()
    #     dist = sampler.run(circuit, shots=10000).result().quasi_dists[0]

    # print(dist)
    # fig = plot_distribution(dist)
    # plt.savefig("simon_distribution.png")
    
    while len(equations) < len(s):
        current_eqn = run_simon_circuit(simon_circuit(s), 1)
        # print(current_eqn)
        if len(current_eqn) == 0:
            continue
        # Check linear independence and other conditions
        if current_eqn[0] not in equations and current_eqn[0] != '0' * len(s) and is_independent(current_eqn[0], A):
            equations.extend(current_eqn)
            print(equations)
            A = np.vstack([A, np.array([int(bit) for bit in current_eqn[0]])])

    print("Equations:")
    print(A)

    # Solve the system of equations
    sym_A = Matrix(A)
    b = sym_A.solve(Matrix([0] * len(s)))
    print("Secret string:")
    print(b)

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
