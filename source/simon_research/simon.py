from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import Operator
import numpy as np
from sympy import Matrix, GF

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
    print(simon_circuit)
    return simon_circuit

def run_simon_circuit(simon_circuit, shots=1024):
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
    new_row = np.array(new_equation).reshape(1, -1)
    augmented_matrix = np.vstack([matrix, new_row])
    rank_before = np.linalg.matrix_rank(matrix, tol=None)
    rank_after = np.linalg.matrix_rank(augmented_matrix, tol=None)
    return rank_after > rank_before

def _iszero(x):
    """
    Returns True if x is zero.
    """
    return getattr(x, 'is_zero', None)

def _nullspace(M, simplify=False, iszerofunc=_iszero):
    """
    Returns list of vectors (Matrix objects) that span nullspace of ``M``

    Examples
    ========

    >>> from sympy import Matrix
    >>> M = Matrix(3, 3, [1, 3, 0, -2, -6, 0, 3, 9, 6])
    >>> M
    Matrix([
    [ 1,  3, 0],
    [-2, -6, 0],
    [ 3,  9, 6]])
    >>> M.nullspace()
    [Matrix([
    [-3],
    [ 1],
    [ 0]])]

    See Also
    ========

    columnspace
    rowspace
    """

    reduced, pivots = M.rref(iszerofunc=iszerofunc, simplify=simplify)

    free_vars = [i for i in range(M.cols) if i not in pivots]
    basis     = []

    for free_var in free_vars:
        # for each free variable, we will set it to 1 and all others
        # to 0.  Then, we will use back substitution to solve the system
        vec           = [M.zero] * M.cols
        vec[free_var] = M.one

        for piv_row, piv_col in enumerate(pivots):
            vec[piv_col] -= reduced[piv_row, free_var]

        basis.append(vec)

    return [M._new(M.cols, 1, b) for b in basis]

if __name__ == "__main__":
    s = '00011'
    equations = run_simon_circuit(simon_circuit(s), shots=1024)

    A = np.empty((0, len(s)), int)
    for bitstring in equations:
        new_equation = np.array([int(bit) for bit in bitstring])
        if is_independent(new_equation, A):
            A = np.vstack([A, new_equation])

    print("Equations:")
    print(equations)
    print("Independent Equations Matrix:")
    print(A)

    A_sympy = Matrix(A)
    nullspace = _nullspace(A_sympy)
    print("Nullspace:")
    print(nullspace)
