"""
Cryptographic Applications using Optimized GF(2) Operations
==========================================================

This demonstrates how the bitwise GF(2) optimizations can be applied to:
1. LFSR (Linear Feedback Shift Register) analysis and attacks
2. Linear cryptanalysis of block ciphers  
3. Stream cipher cryptanalysis
4. Hash function analysis

Applications:
- Breaking weak stream ciphers
- Cryptanalysis of LFSR-based generators
- Linear attacks on block ciphers
- Analysis of hash functions with linear components
"""

import numpy as np
import random
import time
from gf2_solver import (pack_vector, unpack_vector, gaussian_elimination_GF2_bitwise,
                        get_secret_integer_bitwise, get_secret_integer_generic)


class LFSRAnalyzer:
    """
    LFSR (Linear Feedback Shift Register) analyzer using optimized GF(2) operations.
    Used for breaking LFSR-based stream ciphers and PRNGs.
    """

    def __init__(self, degree):
        """Initialize analyzer for LFSR of given degree."""
        self.degree = degree

    def berlekamp_massey_bitwise(self, sequence):
        """
        Berlekamp-Massey algorithm using bitwise operations to find the
        shortest LFSR that generates the given sequence.
        
        This is crucial for cryptanalysis of stream ciphers.
        """
        start_time = time.time()

        n = len(sequence)
        # Convert sequence to packed integers for faster processing
        if n <= 64:  # Can fit in a single 64-bit integer
            seq_packed = pack_vector(sequence)

        # Initialize
        c = [0] * n  # connection polynomial
        b = [0] * n  # auxiliary polynomial
        c[0] = b[0] = 1
        l, m, d = 0, -1, 1

        for i in range(n):
            # Compute discrepancy using bitwise operations
            discrepancy = sequence[i]
            for j in range(1, l + 1):
                discrepancy ^= (c[j] & sequence[i - j])

            if discrepancy == 1:
                temp = c[:]
                for j in range(n - i - 1):
                    if j + i - m < n:
                        c[j + i - m] ^= b[j]

                if 2 * l <= i:
                    l = i + 1 - l
                    m = i
                    b = temp

            d <<= 1
            d &= ((1 << n) - 1)  # Keep within n bits

        elapsed = time.time() - start_time

        # Extract the minimal polynomial
        minimal_poly = c[:l + 1]
        return minimal_poly, l, elapsed

    def known_plaintext_attack_bitwise(self, ciphertext, plaintext):
        """
        Known plaintext attack on LFSR-based stream cipher.
        Recovers the keystream and potentially the LFSR structure.
        """
        if len(ciphertext) != len(plaintext):
            raise ValueError("Ciphertext and plaintext must have same length")

        start_time = time.time()

        # Recover keystream by XORing ciphertext with plaintext
        keystream = []
        for i in range(len(ciphertext)):
            keystream.append(ciphertext[i] ^ plaintext[i])

        # Use Berlekamp-Massey to find the generating LFSR
        minimal_poly, degree, _ = self.berlekamp_massey_bitwise(keystream)

        elapsed = time.time() - start_time
        return keystream, minimal_poly, degree, elapsed


class LinearCryptanalysis:
    """
    Linear cryptanalysis tools using optimized GF(2) operations.
    Used for analyzing block ciphers and finding linear approximations.
    """

    def __init__(self, block_size):
        """Initialize for given block size."""
        self.block_size = block_size

    def find_linear_approximations_bitwise(self, sbox, input_mask, output_mask):
        """
        Find linear approximations in S-boxes using bitwise operations.
        Critical for analyzing the security of block ciphers.
        """
        start_time = time.time()

        sbox_size = len(sbox)
        bias_count = 0

        # Pack masks for faster bitwise operations
        input_mask_packed = pack_vector([int(b) for b in format(input_mask, f'0{sbox_size.bit_length()}b')])
        output_mask_packed = pack_vector([int(b) for b in format(output_mask, f'0{sbox_size.bit_length()}b')])

        for x in range(sbox_size):
            y = sbox[x]

            # Compute input parity using bitwise AND and popcount
            input_parity = bin(x & input_mask).count('1') % 2

            # Compute output parity
            output_parity = bin(y & output_mask).count('1') % 2

            # Check if linear approximation holds
            if input_parity == output_parity:
                bias_count += 1

        # Calculate bias
        bias = abs(bias_count - sbox_size // 2)
        probability = bias_count / sbox_size

        elapsed = time.time() - start_time
        return bias, probability, elapsed

    def correlation_attack_bitwise(self, keystream_samples, correlation_matrix):
        """
        Correlation attack using optimized matrix operations over GF(2).
        Used to break combination generators and filter generators.
        """
        start_time = time.time()

        # Convert correlation matrix to list format for our solver
        matrix_list = correlation_matrix.tolist() if isinstance(correlation_matrix,
                                                                np.ndarray) else correlation_matrix

        # Use our optimized GF(2) solver to find correlations
        solution, solve_time = get_secret_integer_bitwise(matrix_list)

        elapsed = time.time() - start_time
        return solution, elapsed


class StreamCipherAnalyzer:
    """
    Stream cipher analyzer using optimized GF(2) operations.
    """

    def __init__(self):
        pass

    def algebraic_attack_bitwise(self, equations_matrix, constants_vector):
        """
        Algebraic attack using Gaussian elimination over GF(2).
        Used to solve polynomial systems arising from stream cipher cryptanalysis.
        """
        start_time = time.time()

        # Augment matrix with constants
        augmented = []
        for i, row in enumerate(equations_matrix):
            augmented_row = row + [constants_vector[i]]
            augmented.append(augmented_row)

        # Use our optimized solver
        try:
            solution, solve_time = get_secret_integer_bitwise(equations_matrix)
            success = True
        except:
            solution = None
            success = False

        elapsed = time.time() - start_time
        return solution, success, elapsed

    def guess_and_determine_attack_bitwise(self, cipher_equations, known_bits, guess_positions):
        """
        Guess-and-determine attack using fast GF(2) solving.
        Try different guesses for unknown bits and solve the resulting system.
        """
        start_time = time.time()

        num_guesses = 2**len(guess_positions)
        solutions = []

        for guess in range(num_guesses):
            # Create modified equations with guessed values
            modified_equations = [row[:] for row in cipher_equations]

            # Substitute guessed values
            guess_bits = [(guess >> i) & 1 for i in range(len(guess_positions))]
            for i, pos in enumerate(guess_positions):
                for row in modified_equations:
                    if len(row) > pos:
                        # If this position has a 1, XOR with the guessed bit
                        if row[pos] == 1:
                            row[-1] ^= guess_bits[i]  # Assuming last column is constants
                        row[pos] = 0  # Set to 0 since we've substituted

            # Try to solve with this guess
            try:
                solution, _ = get_secret_integer_bitwise(modified_equations)
                # Reconstruct full solution including guessed bits
                full_solution = solution
                solutions.append((guess_bits, full_solution))
            except:
                continue

        elapsed = time.time() - start_time
        return solutions, elapsed


def benchmark_crypto_applications():
    """Benchmark cryptographic applications."""
    print("=== Cryptographic Applications Benchmark ===")

    # 1. LFSR Analysis
    print("\n1. LFSR Analysis:")
    lfsr_analyzer = LFSRAnalyzer(degree=16)

    # Generate a test sequence from a known LFSR
    test_sequence = [random.randint(0, 1) for _ in range(100)]

    poly, degree, lfsr_time = lfsr_analyzer.berlekamp_massey_bitwise(test_sequence)
    print(f"   Berlekamp-Massey on 100-bit sequence: {lfsr_time*1000:.3f} ms")
    print(f"   Found LFSR of degree: {degree}")

    # 2. Linear Cryptanalysis
    print("\n2. Linear Cryptanalysis:")
    lin_crypto = LinearCryptanalysis(block_size=64)

    # Example S-box (simplified)
    sbox = list(range(16))
    random.shuffle(sbox)

    bias, prob, lin_time = lin_crypto.find_linear_approximations_bitwise(sbox, 0b1010, 0b0110)
    print(f"   Linear approximation analysis: {lin_time*1000:.3f} ms")
    print(f"   Bias: {bias}, Probability: {prob:.3f}")

    # 3. Stream Cipher Analysis
    print("\n3. Stream Cipher Algebraic Attack:")
    stream_analyzer = StreamCipherAnalyzer()

    # Generate a small system of equations
    n = 50
    equations = [[random.randint(0, 1) for _ in range(n)] for _ in range(n - 1)]

    solution, success, solve_time = stream_analyzer.algebraic_attack_bitwise(equations, [0] * (n - 1))
    print(f"   Algebraic attack on {n}x{n} system: {solve_time*1000:.3f} ms")
    print(f"   Success: {success}")


def demo_lfsr_attack():
    """Demonstrate a complete LFSR attack scenario."""
    print("\n=== LFSR Attack Demonstration ===")

    # Simulate a simple LFSR-based stream cipher
    def simple_lfsr(seed, taps, length):
        """Generate LFSR sequence."""
        state = seed
        sequence = []

        for _ in range(length):
            bit = state & 1
            sequence.append(bit)

            # Compute feedback
            feedback = 0
            for tap in taps:
                feedback ^= (state >> tap) & 1

            state = (state >> 1) | (feedback << (max(taps)))

        return sequence

    # Generate keystream from a weak LFSR
    seed = 0b1101010110111  # 13-bit seed
    taps = [0, 1, 3, 4]  # Feedback taps
    keystream_length = 200

    keystream = simple_lfsr(seed, taps, keystream_length)
    print(f"Generated {keystream_length}-bit keystream from LFSR")

    # Simulate known plaintext attack
    plaintext = [random.randint(0, 1) for _ in range(keystream_length)]
    ciphertext = [p ^ k for p, k in zip(plaintext, keystream)]

    # Attack the cipher
    analyzer = LFSRAnalyzer(degree=len(format(seed, 'b')))
    recovered_keystream, recovered_poly, recovered_degree, attack_time = analyzer.known_plaintext_attack_bitwise(
        ciphertext, plaintext)

    print(f"Attack completed in {attack_time*1000:.3f} ms")
    print(f"Recovered LFSR degree: {recovered_degree}")
    print(f"Original keystream matches recovered: {keystream[:50] == recovered_keystream[:50]}")


if __name__ == "__main__":
    # Run benchmarks
    benchmark_crypto_applications()

    # Run LFSR attack demo
    demo_lfsr_attack()
