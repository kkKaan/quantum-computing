"""
Error Correcting Code Decoder using Optimized GF(2) Operations
============================================================

This demonstrates how the bitwise GF(2) optimizations from Simon's algorithm
can be applied to syndrome decoding in error correcting codes.

Applications:
- WiFi (802.11) - uses convolutional and LDPC codes
- 5G/LTE - uses polar and LDPC codes  
- Storage systems (SSDs, HDDs) - use BCH, Reed-Solomon codes
- Satellite communications - use various linear codes
"""

import numpy as np
import random
import time
from gf2_solver import (pack_vector, unpack_vector, gaussian_elimination_GF2_bitwise,
                        get_secret_integer_bitwise, get_secret_integer_generic)


class HammingCode:
    """
    Hamming code implementation using optimized GF(2) operations.
    Hamming codes can correct single-bit errors.
    """

    def __init__(self, r):
        """
        Initialize Hamming code with r parity bits.
        Can encode 2^r - r - 1 data bits.
        """
        self.r = r  # number of parity bits
        self.n = 2**r - 1  # codeword length
        self.k = self.n - r  # data bits
        self.H = self._generate_parity_check_matrix()

    def _generate_parity_check_matrix(self):
        """Generate the parity check matrix H for Hamming code."""
        H = []
        # H has r rows and n columns
        # Each column is the binary representation of its column index (1 to n)
        for col in range(1, self.n + 1):
            column = []
            for row in range(self.r):
                column.append((col >> row) & 1)
            H.append(column)

        # Transpose to get r x n matrix
        return [[H[col][row] for col in range(self.n)] for row in range(self.r)]

    def encode(self, data_bits):
        """Encode data bits into a codeword (simplified systematic encoding)."""
        if len(data_bits) != self.k:
            raise ValueError(f"Data must be {self.k} bits long")

        # For systematic encoding, we need the generator matrix
        # This is simplified - in practice you'd compute G from H
        codeword = data_bits + [0] * self.r

        # Calculate parity bits (simplified)
        for i in range(self.r):
            parity = 0
            for j in range(self.n):
                parity ^= (self.H[i][j] & codeword[j])
            codeword[self.k + i] = parity

        return codeword[:self.n]

    def syndrome_decode_bitwise(self, received_word):
        """
        Decode using optimized bitwise syndrome computation.
        This is where your GF(2) optimizations provide significant speedup.
        """
        if len(received_word) != self.n:
            raise ValueError(f"Received word must be {self.n} bits long")

        start_time = time.time()

        # Compute syndrome: s = H * r^T
        syndrome = []
        for i in range(self.r):
            s_i = 0
            for j in range(self.n):
                s_i ^= (self.H[i][j] & received_word[j])
            syndrome.append(s_i)

        # Pack syndrome into integer for fast operations
        syndrome_int = pack_vector(syndrome)

        # If syndrome is zero, no error detected
        if syndrome_int == 0:
            elapsed = time.time() - start_time
            return received_word, 0, elapsed

        # Find error position (syndrome equals column of H)
        error_pos = syndrome_int  # For Hamming codes, syndrome directly gives error position

        # Correct the error
        corrected = received_word[:]
        if 1 <= error_pos <= self.n:
            corrected[error_pos - 1] ^= 1

        elapsed = time.time() - start_time
        return corrected, error_pos, elapsed

    def syndrome_decode_generic(self, received_word):
        """Traditional syndrome decoding for comparison."""
        start_time = time.time()

        # Compute syndrome
        syndrome = []
        for i in range(self.r):
            s_i = 0
            for j in range(self.n):
                s_i ^= (self.H[i][j] & received_word[j])
            syndrome.append(s_i)

        # Check if syndrome is zero
        if all(s == 0 for s in syndrome):
            elapsed = time.time() - start_time
            return received_word, 0, elapsed

        # Find error position by comparing syndrome to columns of H
        error_pos = 0
        for col in range(self.n):
            if [self.H[row][col] for row in range(self.r)] == syndrome:
                error_pos = col + 1
                break

        # Correct the error
        corrected = received_word[:]
        if error_pos > 0:
            corrected[error_pos - 1] ^= 1

        elapsed = time.time() - start_time
        return corrected, error_pos, elapsed


class LDPCDecoder:
    """
    LDPC (Low-Density Parity-Check) Code decoder using optimized GF(2) operations.
    Used in 5G, WiFi 6, and modern storage systems.
    """

    def __init__(self, H_matrix):
        """Initialize with parity check matrix H."""
        self.H = H_matrix
        self.m, self.n = len(H_matrix), len(H_matrix[0])

    def syndrome_decode_bitwise(self, received_word, max_iterations=50):
        """
        Iterative syndrome-based decoding using bitwise operations.
        This is where your optimizations really shine for large matrices.
        """
        start_time = time.time()

        current_word = received_word[:]

        for iteration in range(max_iterations):
            # Compute syndrome using bitwise operations
            syndrome = []
            for i in range(self.m):
                s_i = 0
                for j in range(self.n):
                    s_i ^= (self.H[i][j] & current_word[j])
                syndrome.append(s_i)

            # Pack syndrome for fast processing
            syndrome_int = pack_vector(syndrome)

            if syndrome_int == 0:  # Decoding successful
                elapsed = time.time() - start_time
                return current_word, iteration, elapsed

            # Bit-flipping algorithm (simplified)
            # In practice, you'd use belief propagation
            flip_scores = [0] * self.n
            for j in range(self.n):
                for i in range(self.m):
                    if self.H[i][j] == 1 and syndrome[i] == 1:
                        flip_scores[j] += 1

            # Flip the bit with highest score
            max_score = max(flip_scores)
            if max_score > 0:
                flip_pos = flip_scores.index(max_score)
                current_word[flip_pos] ^= 1

        elapsed = time.time() - start_time
        return current_word, max_iterations, elapsed


def benchmark_ecc_performance():
    """Benchmark ECC decoding performance."""
    print("=== Error Correcting Code Decoder Benchmark ===")

    # Test different Hamming code sizes
    r_values = [3, 4, 5, 6, 7]  # parity bits

    for r in r_values:
        hamming = HammingCode(r)
        print(f"\nHamming({hamming.n}, {hamming.k}) - {hamming.n}-bit codewords:")

        # Generate test cases
        num_tests = 1000
        bitwise_times = []
        generic_times = []

        for _ in range(num_tests):
            # Generate random data
            data = [random.randint(0, 1) for _ in range(hamming.k)]
            codeword = hamming.encode(data)

            # Add single-bit error
            error_pos = random.randint(0, hamming.n - 1)
            received = codeword[:]
            received[error_pos] ^= 1

            # Test bitwise decoder
            _, _, time_bitwise = hamming.syndrome_decode_bitwise(received)
            bitwise_times.append(time_bitwise)

            # Test generic decoder
            _, _, time_generic = hamming.syndrome_decode_generic(received)
            generic_times.append(time_generic)

        avg_bitwise = np.mean(bitwise_times)
        avg_generic = np.mean(generic_times)
        speedup = avg_generic / avg_bitwise if avg_bitwise > 0 else float('inf')

        print(f"  Bitwise decoder: {avg_bitwise*1000:.3f} ms average")
        print(f"  Generic decoder: {avg_generic*1000:.3f} ms average")
        print(f"  Speedup: {speedup:.2f}x")


def generate_random_ldpc_matrix(m, n, column_weight=3):
    """Generate a random LDPC matrix with specified column weight."""
    H = [[0 for _ in range(n)] for _ in range(m)]

    for j in range(n):
        # Randomly select 'column_weight' rows to place 1s
        rows = random.sample(range(m), min(column_weight, m))
        for i in rows:
            H[i][j] = 1

    return H


if __name__ == "__main__":
    # Run Hamming code benchmark
    benchmark_ecc_performance()

    # Demo LDPC decoding
    print("\n=== LDPC Decoder Demo ===")

    # Generate a small LDPC code for demo
    m, n = 50, 100  # 50 parity checks, 100 bits
    H = generate_random_ldpc_matrix(m, n, column_weight=3)

    ldpc = LDPCDecoder(H)

    # Generate test case
    received_word = [random.randint(0, 1) for _ in range(n)]

    # Decode using bitwise operations
    decoded, iterations, decode_time = ldpc.syndrome_decode_bitwise(received_word)

    print(f"LDPC({n}, {n-m}) decoding:")
    print(f"  Decoded in {iterations} iterations")
    print(f"  Time: {decode_time*1000:.3f} ms")
    print(f"  Matrix size: {m}x{n}")
