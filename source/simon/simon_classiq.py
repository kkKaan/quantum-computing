from classiq import *
import galois
import numpy as np
from classiq.qmod.symbolic import min
import matplotlib.pyplot as plt
from classiq.execution import ExecutionPreferences


@qfunc
def simon_qfunc(f_qfunc: QCallable[QNum, Output[QNum]], x: QNum):
    res = QNum("res")
    hadamard_transform(x)
    f_qfunc(x, res)
    hadamard_transform(x)


# here we work over boolean arithmetics - F(2)
GF = galois.GF(2)

# The following function checks whether a set contains linearly independet vectors


def is_independent_set(vectors):
    matrix = GF(vectors)
    rank = np.linalg.matrix_rank(matrix)
    if rank == len(vectors):
        return True
    else:
        return False


def get_independent_set(samples):
    """
    The following function gets samples of n-sized strings from running the quantum part and return an n-1 x n matrix,
    whose rows forms a set if independent
    """
    ind_v = []
    for v in samples:
        if is_independent_set(ind_v + [v]):
            ind_v.append(v)
            if len(ind_v) == len(v) - 1:
                # reached max set of N-1
                break
    return ind_v


def get_secret_integer(matrix):
    gf_v = GF(matrix)  # converting to a matrix over Z_2
    null_space = gf_v.T.left_null_space()  # finding the right-null space of the matrix
    return int("".join(np.array(null_space)[0][::-1].astype(str)), 2)  # converting from binary to integer


@qfunc
def simon_qfunc_simple(s: CInt, x: QNum, res: Output[QNum]):
    res |= min(x, x ^ s)


NUM_QUBITS = 5
S_SECRET = 6


@qfunc
def main(x: Output[QNum], res: Output[QNum]):
    allocate(NUM_QUBITS, x)
    hadamard_transform(x)
    simon_qfunc_simple(S_SECRET, x, res)


qmod = create_model(main)
qmod = update_constraints(qmod, optimization_parameter="width")

# synthesize
qprog = synthesize(qmod)
# vizualize
show(qprog)

# execute
result = execute(qprog).result_value()

my_result = {sample.state["x"]: sample.state["res"] for sample in result.parsed_counts}
fig, ax = plt.subplots()
ax.plot(my_result.keys(), my_result.values(), "o")
ax.grid(axis="y", which="minor")
ax.grid(axis="y", which="major")
ax.grid(axis="x", which="minor")
ax.grid(axis="x", which="major")
plt.xlabel("$x$", fontsize=16)
plt.ylabel("$f(x)$", fontsize=16)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
ax.minorticks_on()


@qfunc
def main(x: Output[QNum]):
    allocate(NUM_QUBITS, x)
    simon_qfunc(lambda x, res: simon_qfunc_simple(S_SECRET, x, res), x)


qmod = create_model(
    main,
    constraints=Constraints(optimization_parameter="width"),
    execution_preferences=ExecutionPreferences(num_shots=50 * NUM_QUBITS),
    out_file="simon_example",
)

qprog = synthesize(qmod)
result = execute(qprog).result_value()
samples = [[int(k) for k in key] for key in result.counts_of_output("x").keys()]

matrix_of_ind_v = get_independent_set(samples)
assert (len(matrix_of_ind_v) == NUM_QUBITS -
        1), "Failed to find an independent set, try to increase the number of shots"
quantum_secret_integer = get_secret_integer(matrix_of_ind_v)

print("The secret binary string (integer) of f(x):", S_SECRET)
print("The result of the Simon's Algorithm:", quantum_secret_integer)
assert (S_SECRET == quantum_secret_integer), "The Simon's algorithm failed to find the secret key."
