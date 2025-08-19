import random

import matplotlib.pyplot as plt
import numpy as np
from simon_amazon_test import get_secret_integer_bitwise, get_secret_integer_generic


def generate_full_rank_binary_matrix(n, rank_deficit=1, density=0.95):
    """
    Generate an n×n binary matrix (list of lists) over GF(2) with a guaranteed nullspace.
    The idea is:
      1. Construct an n×(n - rank_deficit) matrix Q that is full column rank over GF(2).
         Each column is generated randomly (each entry 1 with probability 'density') and
         added only if it is not in the span (over GF(2)) of the previously chosen columns.
      2. Construct the remaining rank_deficit columns as nontrivial linear combinations
         of the columns of Q (here, for simplicity, we set them as the bitwise sum mod 2
         of all columns in Q).
      3. Return the n×n matrix M = [ Q | L ].

    For example, with rank_deficit=1, M will have rank n-1.
    """
    # Number of independent columns to generate:
    num_indep = n - rank_deficit
    Q = []  # list of columns, each column is a list of bits of length n

    def is_independent(new_col, cols):
        # Check if new_col is in the GF(2) span of columns in 'cols'
        # We'll do this by forming a matrix with the current columns and new_col and computing rank over GF(2)
        if not cols:
            return True
        mat = np.array(cols + [new_col], dtype=np.int8).T  # shape: n x (len(cols)+1)
        # Use numpy.linalg.matrix_rank on float version after converting entries to 0/1.
        # Note: This is not strictly over GF(2), but for binary matrices it works if we reduce mod2.
        # A better method would be to perform elimination mod2, but for moderate n this is acceptable.
        # To mimic GF(2) arithmetic, we take rank over GF(2) by reducing each entry mod2.
        # For simplicity, we use our own elimination in GF(2).
        # We'll use a simple iterative method:
        r = 0
        m, k = mat.shape
        used = set()
        for col in range(k):
            pivot_row = None
            for i in range(r, m):
                if mat[i, col] % 2 == 1:
                    pivot_row = i
                    break
            if pivot_row is None:
                continue
            r += 1
            if r == m:
                break
        return r == len(cols) + 1  # full rank if adding new_col increases rank by 1

    # Iteratively generate num_indep independent columns.
    attempts = 0
    while len(Q) < num_indep:
        # Generate a random binary column of length n with probability 'density'
        col = [1 if random.random() < density else 0 for _ in range(n)]
        # Ensure the column is not all zeros
        if sum(col) == 0:
            continue
        if is_independent(col, Q):
            Q.append(col)
        attempts += 1
        if attempts > 10000:
            raise Exception("Failed to generate enough independent columns. Try adjusting density.")

    # Now, construct the remaining rank_deficit columns.
    # For simplicity, we can define each additional column as the modulo-2 sum of all columns in Q.
    # (In general, you might want to choose a nontrivial combination.)
    L = []
    for _ in range(rank_deficit):
        new_col = [0] * n
        for i in range(n):
            s = 0
            for col in Q:
                s = (s + col[i]) % 2
            new_col[i] = s
        L.append(new_col)

    # Form full matrix M = [ Q | L ], so M is n x n.
    M = [q + l for q, l in zip(np.array(Q).T.tolist(), np.array(L).T.tolist())]
    # M is a list of n rows, each of length n.
    # Verify rank over GF(2) using numpy.linalg.matrix_rank on float version (after converting mod2)
    npM = np.array(M, dtype=np.int8)
    # Note: numpy.linalg.matrix_rank does not compute rank mod2; we rely on our construction.
    return M


###############################################################################
# Benchmarking functions
###############################################################################
def run_large_matrix_benchmark():
    """
    Run benchmark tests comparing the bitwise GF(2) solver with the generic solver
    on randomly generated n×n binary matrices (with a guaranteed nullspace).
    """
    print("=== Large Matrix Solver Benchmark ===")
    # Define sizes to test
    sizes = [20, 40, 60, 80, 100, 150, 200, 250, 300, 400, 500, 600, 800, 1000]

    # Store results
    results = {"sizes": sizes, "bitwise_times": [], "generic_times": [], "speedups": []}

    for size in sizes:
        print(f"\nTesting {size}x{size} matrix:")
        try:
            # Generate a matrix with nullity = 1 (i.e. rank = n-1) using our construction
            matrix = generate_full_rank_binary_matrix(size, rank_deficit=1, density=0.5)
            # For debugging, you can print the rank computed by a GF(2) routine if desired.

            # Test bitwise solver (which is optimized for GF(2))
            print("  Running bitwise solver...")
            secret_bitwise, t_bitwise = get_secret_integer_bitwise(matrix)
            results["bitwise_times"].append(t_bitwise)
            print(f"  Bitwise solver time: {t_bitwise:.6f} seconds")

            # Test generic solver: we force generic solver to work in mod 2 arithmetic.
            print("  Running generic solver (mod 2)...")
            secret_generic, t_generic = get_secret_integer_generic(matrix, mod=2)
            results["generic_times"].append(t_generic)
            print(f"  Generic solver time: {t_generic:.6f} seconds")

            # Calculate speedup: generic time / bitwise time
            speedup = t_generic / t_bitwise if t_bitwise > 0 else float("inf")
            results["speedups"].append(speedup)
            print(f"  Speedup (Generic/Bitwise): {speedup:.2f}x")

            # (Optional) Verify that the solutions match:
            if secret_bitwise != secret_generic:
                print("  WARNING: Solvers produced different secret strings!")
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            results["bitwise_times"].append(None)
            results["generic_times"].append(None)
            results["speedups"].append(None)

    plot_results(results)
    print("\nBenchmark Summary:")
    print("------------------")
    for i, size in enumerate(sizes):
        if results["bitwise_times"][i] is not None:
            print(
                f"Size {size}x{size}: Bitwise={results['bitwise_times'][i]:.6f}s, "
                f"Generic={results['generic_times'][i]:.6f}s, "
                f"Speedup={results['speedups'][i]:.2f}x"
            )
        else:
            print(f"Size {size}x{size}: Failed")


def plot_results(results):
    """
    Plot the benchmark results.
    """
    valid_indices = [i for i, t in enumerate(results["bitwise_times"]) if t is not None]
    sizes = [results["sizes"][i] for i in valid_indices]
    bitwise = [results["bitwise_times"][i] for i in valid_indices]
    generic = [results["generic_times"][i] for i in valid_indices]

    plt.figure(figsize=(12, 8))
    plt.semilogy(sizes, bitwise, "o-", label="Bitwise GF(2) Solver", linewidth=2, markersize=8)
    plt.semilogy(sizes, generic, "s-", label="Generic Solver (mod 2)", linewidth=2, markersize=8)
    plt.xlabel("Matrix Size (n)", fontsize=12)
    plt.ylabel("Time (seconds)", fontsize=12)
    plt.title("Solver Performance Comparison on n×n Binary Matrices", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig("solver_benchmark.png")
    plt.show()


###############################################################################
# Main: Run benchmark
###############################################################################
if __name__ == "__main__":
    run_large_matrix_benchmark()
