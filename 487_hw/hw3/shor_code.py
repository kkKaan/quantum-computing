from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from numpy import pi

# q2.2
qreg_q = QuantumRegister(1, 'q')
creg_c = ClassicalRegister(1, 'c')
circuit = QuantumCircuit(qreg_q, creg_c)

circuit.reset(qreg_q[0])
circuit.h(qreg_q[0])
circuit.p(46 * pi / 33, qreg_q[0])
circuit.h(qreg_q[0])
circuit.measure(qreg_q[0], creg_c[0])
# @columns [0,1,2,3,4]


# q2.3
qreg_q = QuantumRegister(9, 'q')
creg_c = ClassicalRegister(1, 'c')
circuit = QuantumCircuit(qreg_q, creg_c)

circuit.reset(qreg_q[0])
circuit.reset(qreg_q[1])
circuit.reset(qreg_q[2])
circuit.reset(qreg_q[3])
circuit.reset(qreg_q[4])
circuit.reset(qreg_q[5])
circuit.reset(qreg_q[6])
circuit.reset(qreg_q[7])
circuit.reset(qreg_q[8])
circuit.h(qreg_q[0])
circuit.p(46 * pi / 33, qreg_q[0])
circuit.h(qreg_q[0])
circuit.barrier(qreg_q[0])
circuit.barrier(qreg_q[1])
circuit.barrier(qreg_q[2])
circuit.barrier(qreg_q[3])
circuit.barrier(qreg_q[4])
circuit.barrier(qreg_q[5])
circuit.barrier(qreg_q[6])
circuit.barrier(qreg_q[7])
circuit.barrier(qreg_q[8])
circuit.cx(qreg_q[0], qreg_q[3])
circuit.cx(qreg_q[0], qreg_q[6])
circuit.h(qreg_q[0])
circuit.h(qreg_q[3])
circuit.h(qreg_q[6])
circuit.cx(qreg_q[0], qreg_q[1])
circuit.cx(qreg_q[3], qreg_q[4])
circuit.cx(qreg_q[6], qreg_q[7])
circuit.cx(qreg_q[0], qreg_q[2])
circuit.cx(qreg_q[3], qreg_q[5])
circuit.cx(qreg_q[6], qreg_q[8])
circuit.barrier(qreg_q[0])
circuit.barrier(qreg_q[1])
circuit.barrier(qreg_q[2])
circuit.barrier(qreg_q[3])
circuit.barrier(qreg_q[4])
circuit.barrier(qreg_q[5])
circuit.barrier(qreg_q[6])
circuit.barrier(qreg_q[7])
circuit.barrier(qreg_q[8])
circuit.x(qreg_q[0])
circuit.z(qreg_q[0])
circuit.barrier(qreg_q[0])
circuit.barrier(qreg_q[1])
circuit.barrier(qreg_q[2])
circuit.barrier(qreg_q[3])
circuit.barrier(qreg_q[4])
circuit.barrier(qreg_q[5])
circuit.barrier(qreg_q[6])
circuit.barrier(qreg_q[7])
circuit.barrier(qreg_q[8])
circuit.cx(qreg_q[0], qreg_q[1])
circuit.cx(qreg_q[3], qreg_q[4])
circuit.cx(qreg_q[6], qreg_q[7])
circuit.cx(qreg_q[0], qreg_q[2])
circuit.cx(qreg_q[3], qreg_q[5])
circuit.cx(qreg_q[6], qreg_q[8])
circuit.ccx(qreg_q[2], qreg_q[1], qreg_q[0])
circuit.ccx(qreg_q[5], qreg_q[4], qreg_q[3])
circuit.ccx(qreg_q[8], qreg_q[7], qreg_q[6])
circuit.h(qreg_q[0])
circuit.h(qreg_q[3])
circuit.h(qreg_q[6])
circuit.cx(qreg_q[0], qreg_q[3])
circuit.cx(qreg_q[0], qreg_q[6])
circuit.ccx(qreg_q[6], qreg_q[3], qreg_q[0])
circuit.barrier(qreg_q[0])
circuit.barrier(qreg_q[1])
circuit.barrier(qreg_q[2])
circuit.barrier(qreg_q[3])
circuit.barrier(qreg_q[4])
circuit.barrier(qreg_q[5])
circuit.barrier(qreg_q[6])
circuit.barrier(qreg_q[7])
circuit.barrier(qreg_q[8])
circuit.measure(qreg_q[0], creg_c[0])
# @columns [0,0,0,0,0,0,0,0,0,1,2,3,4,4,4,4,4,4,4,4,4,5,6,7,7,7,8,8,8,9,9,9,10,10,10,10,10,10,10,10,10,11,12,13,13,13,13,13,13,13,13,13,14,14,14,15,15,15,16,16,16,17,17,17,18,19,20,21,21,21,21,21,21,21,21,21,22]


# from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
# from numpy import pi

def create_circuit_q22():
    qreg_q = QuantumRegister(1, 'q')
    creg_c = ClassicalRegister(1, 'c')
    circuit = QuantumCircuit(qreg_q, creg_c)

    circuit.reset(qreg_q[0])
    circuit.h(qreg_q[0])
    circuit.p(46 * pi / 33, qreg_q[0])
    circuit.h(qreg_q[0])
    circuit.measure(qreg_q[0], creg_c[0])
    
    return circuit

def create_circuit_q23():
    qreg_q = QuantumRegister(9, 'q')
    creg_c = ClassicalRegister(1, 'c')
    circuit = QuantumCircuit(qreg_q, creg_c)

    # Reset all qubits
    for i in range(9):
        circuit.reset(qreg_q[i])

    circuit.h(qreg_q[0])
    circuit.p(46 * pi / 33, qreg_q[0])
    circuit.h(qreg_q[0])
    circuit.barrier()

    # Apply CX gates and H gates as per the design
    circuit.cx(qreg_q[0], qreg_q[3])
    circuit.cx(qreg_q[0], qreg_q[6])
    circuit.h(qreg_q[0])
    circuit.h(qreg_q[3])
    circuit.h(qreg_q[6])

    circuit.cx(qreg_q[0], qreg_q[1])
    circuit.cx(qreg_q[3], qreg_q[4])
    circuit.cx(qreg_q[6], qreg_q[7])

    circuit.cx(qreg_q[0], qreg_q[2])
    circuit.cx(qreg_q[3], qreg_q[5])
    circuit.cx(qreg_q[6], qreg_q[8])
    circuit.barrier()

    circuit.x(qreg_q[0])
    circuit.z(qreg_q[0])
    circuit.barrier()

    circuit.cx(qreg_q[0], qreg_q[1])
    circuit.cx(qreg_q[3], qreg_q[4])
    circuit.cx(qreg_q[6], qreg_q[7])

    circuit.cx(qreg_q[0], qreg_q[2])
    circuit.cx(qreg_q[3], qreg_q[5])
    circuit.cx(qreg_q[6], qreg_q[8])

    circuit.ccx(qreg_q[2], qreg_q[1], qreg_q[0])
    circuit.ccx(qreg_q[5], qreg_q[4], qreg_q[3])
    circuit.ccx(qreg_q[8], qreg_q[7], qreg_q[6])

    circuit.h(qreg_q[0])
    circuit.h(qreg_q[3])
    circuit.h(qreg_q[6])

    circuit.cx(qreg_q[0], qreg_q[3])
    circuit.cx(qreg_q[0], qreg_q[6])
    circuit.ccx(qreg_q[6], qreg_q[3], qreg_q[0])
    circuit.barrier()

    # Measure the first qubit
    circuit.measure(qreg_q[0], creg_c[0])
    
    return circuit

# Create and display the circuit
circuit_q23 = create_circuit_q23()
print(circuit_q23.draw())

# Create and display circuit q2.2
circuit_q22 = create_circuit_q22()
print("Circuit q2.2:")
print(circuit_q22.draw())

# Create and display circuit q2.3
circuit_q23 = create_circuit_q23()
print("\nCircuit q2.3:")
print(circuit_q23.draw())
