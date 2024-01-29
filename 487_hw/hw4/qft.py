from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT
from qiskit import Aer, execute
from qiskit.visualization import plot_histogram
import math

# # Your student ID sum is 33, which is 100001 in binary.
# # Create a quantum circuit with 6 qubits.
# qc = QuantumCircuit(6)

# # Initialize the qubits to the binary representation of the sum of the student ID digits.
# # The least significant bit (LSB) is on the right; Qiskit uses LSB on the left.
# # Therefore, we need to reverse the order of the bits to match Qiskit's representation.
# initial_state = '100001'[::-1]  # Reversing the binary string to match Qiskit's order.
# for i, bit in enumerate(initial_state):
#     if bit == '1':
#         qc.x(i)  # Apply the X gate if the bit is 1.

# # Apply Quantum Fourier Transform on all qubits
# qc.append(QFT(len(qc.qubits), do_swaps=False), qc.qubits)

# # Draw the circuit
# qc.draw(output='mpl', filename='qft_circuit.png')

# # Execute the circuit and get the counts
# simulator = Aer.get_backend('qasm_simulator')
# qc.measure_all()
# result = execute(qc, simulator, shots=1024).result()
# counts = result.get_counts(qc)

# # Plot the results
# plot_histogram(counts, filename='qft_results.png')

### Without using Qiskit's QFT library ###

# Function to apply a single-qubit QFT rotation to the circuit
def apply_qft_rotations(circuit, n):
    """Apply QFT rotations to the first n qubits in circuit"""
    if n == 0:
        return circuit
    n -= 1
    circuit.h(n)
    for qubit in range(n):
        circuit.cp(math.pi/2**(n-qubit), qubit, n)
    apply_qft_rotations(circuit, n)

# Create a 6-qubit Quantum Circuit
num_qubits = 6
qc = QuantumCircuit(num_qubits)

# Apply the QFT rotations
apply_qft_rotations(qc, num_qubits)

# For Qiskit, the bits are reversed. Use the binary sum of your student ID '100001'
# Here's the bitstring in the correct order for Qiskit
bitstring = '100001'[::-1]

# Initialize the qubits according to this bitstring
for i, bit in enumerate(bitstring):
    if bit == '1':
        qc.x(i)

# Apply the QFT
apply_qft_rotations(qc, num_qubits)

# Swap the qubits to get them in the right order for the inverse QFT
for i in range(num_qubits//2):
    qc.swap(i, num_qubits-i-1)

# Draw the circuit
circuit_diagram = qc.draw(output='mpl')

# circuit_diagram.show()

circuit_diagram.savefig('qft_circuit.png')