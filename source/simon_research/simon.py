from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import Operator
import numpy as np

def simon_oracle(b):
    """
    Returns a Simon oracle for bitstring b

    :param b: The bitstring for the Simon oracle
    :return: The Simon oracle as a QuantumCircuit
    """
    b = b[::-1] # reverse b for easy iteration
    n = len(b)
    qc = QuantumCircuit(n*2)
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

    return simon_circuit
    

if __name__ == "__main__":
    # Define the string 's' for the Simon oracle
    s = '1010'

    # Create the Simon circuit
    simon_circuit = simon_circuit(s)

    # Print the result
    print(simon_circuit.draw())
