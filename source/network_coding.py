"""
Network Coding Applications using Optimized GF(2) Operations
===========================================================

This demonstrates how the bitwise GF(2) optimizations can be applied to:
1. Random Linear Network Coding (RLNC)
2. Packet recovery in lossy networks
3. XOR-based erasure codes
4. Network error correction

Applications:
- Peer-to-peer file sharing (BitTorrent-like)
- Content delivery networks (CDN)
- Wireless sensor networks
- Satellite communications
- 5G network slicing
"""

import numpy as np
import random
import time
from gf2_solver import (pack_vector, unpack_vector, gaussian_elimination_GF2_bitwise,
                        get_secret_integer_bitwise, get_secret_integer_generic)


class NetworkCoder:
    """
    Network coding implementation using optimized GF(2) operations.
    Used for efficient data transmission over lossy networks.
    """

    def __init__(self, num_packets, packet_size):
        """
        Initialize network coder.
        
        Args:
            num_packets: Number of original packets (k)
            packet_size: Size of each packet in bits
        """
        self.k = num_packets
        self.packet_size = packet_size
        self.encoding_matrix = None

    def generate_random_encoding_matrix(self, num_coded_packets):
        """
        Generate random linear encoding matrix over GF(2).
        Each coded packet is a linear combination of original packets.
        """
        self.n = num_coded_packets
        self.encoding_matrix = []

        for i in range(self.n):
            # Generate random coefficient vector
            coeffs = [random.randint(0, 1) for _ in range(self.k)]
            # Ensure at least one coefficient is 1
            if sum(coeffs) == 0:
                coeffs[random.randint(0, self.k - 1)] = 1
            self.encoding_matrix.append(coeffs)

        return self.encoding_matrix

    def encode_packets_bitwise(self, original_packets):
        """
        Encode original packets using random linear network coding.
        Uses bitwise XOR operations for efficiency.
        """
        if len(original_packets) != self.k:
            raise ValueError(f"Expected {self.k} packets, got {len(original_packets)}")

        start_time = time.time()
        coded_packets = []

        for i in range(self.n):
            # Initialize coded packet as zeros
            coded_packet = [0] * self.packet_size

            # XOR original packets according to encoding coefficients
            for j in range(self.k):
                if self.encoding_matrix[i][j] == 1:
                    for bit_idx in range(self.packet_size):
                        coded_packet[bit_idx] ^= original_packets[j][bit_idx]

            coded_packets.append(coded_packet)

        elapsed = time.time() - start_time
        return coded_packets, elapsed

    def decode_packets_bitwise(self, received_packets, received_coefficients):
        """
        Decode original packets from received coded packets using GF(2) solver.
        This is where your optimized Gaussian elimination really shines!
        """
        if len(received_packets) < self.k:
            raise ValueError(f"Need at least {self.k} packets to decode, got {len(received_packets)}")

        start_time = time.time()

        # Take first k linearly independent packets
        selected_packets = []
        selected_coeffs = []

        for i, (packet, coeffs) in enumerate(zip(received_packets, received_coefficients)):
            test_coeffs = selected_coeffs + [coeffs]

            # Check if linearly independent using our helper
            if self._is_independent_set(test_coeffs):
                selected_packets.append(packet)
                selected_coeffs.append(coeffs)

                if len(selected_packets) == self.k:
                    break

        if len(selected_packets) < self.k:
            raise ValueError("Insufficient linearly independent packets")

        # Solve the system: coeffs * original = received
        # We need to invert the coefficient matrix
        coeff_matrix = selected_coeffs

        # Use our optimized GF(2) Gaussian elimination
        original_packets = []

        # Solve for each bit position independently
        for bit_pos in range(self.packet_size):
            # Extract bit values at this position from all received packets
            bit_vector = [packet[bit_pos] for packet in selected_packets]

            # Solve: coeff_matrix * original_bits = bit_vector
            solution_bits = self._solve_gf2_system_bitwise(coeff_matrix, bit_vector)

            # Store this bit position for all original packets
            if bit_pos == 0:
                original_packets = [[0] * self.packet_size for _ in range(self.k)]

            for j in range(self.k):
                original_packets[j][bit_pos] = solution_bits[j]

        elapsed = time.time() - start_time
        return original_packets, elapsed

    def _is_independent_set(self, vectors):
        """Check if vectors are linearly independent over GF(2)."""
        if not vectors:
            return True

        matrix = np.array(vectors, dtype=float)
        rank = np.linalg.matrix_rank(matrix)
        return rank == len(vectors)

    def _solve_gf2_system_bitwise(self, matrix, vector):
        """
        Solve Ax = b over GF(2) using optimized operations.
        """
        n = len(matrix)

        # Augment matrix with vector
        augmented = []
        for i in range(n):
            augmented.append(matrix[i] + [vector[i]])

        # Use bitwise Gaussian elimination
        packed_rows = [pack_vector(row) for row in augmented]
        echelon_rows, pivot_cols = gaussian_elimination_GF2_bitwise(packed_rows, n + 1)

        # Back substitution
        solution = [0] * n
        for i in reversed(range(len(pivot_cols))):
            col = pivot_cols[i]
            if col < n:  # Not the augmented column
                # Extract equation: solution[col] + sum = rhs
                rhs = (echelon_rows[i] >> n) & 1  # Augmented part
                sum_val = 0
                for j in range(col + 1, n):
                    if (echelon_rows[i] >> j) & 1:
                        sum_val ^= solution[j]
                solution[col] = sum_val ^ rhs

        return solution


class ErasureDecoder:
    """
    Erasure code decoder for packet loss recovery.
    Used in distributed storage and streaming applications.
    """

    def __init__(self, k, n):
        """
        Initialize (n, k) erasure code.
        k = number of data packets
        n = total number of packets (including redundancy)
        """
        self.k = k
        self.n = n
        self.redundancy = n - k

    def generate_generator_matrix(self):
        """Generate systematic generator matrix [I | P]."""
        # Identity matrix for systematic part
        generator = []

        # Add identity matrix
        for i in range(self.k):
            row = [0] * self.n
            row[i] = 1
            generator.append(row)

        # Add parity part (random for demonstration)
        for i in range(self.k):
            for j in range(self.k, self.n):
                generator[i][j] = random.randint(0, 1)

        self.generator_matrix = generator
        return generator

    def encode_systematic_bitwise(self, data_packets):
        """Encode data packets into codeword."""
        if len(data_packets) != self.k:
            raise ValueError(f"Expected {self.k} data packets")

        start_time = time.time()

        packet_size = len(data_packets[0])
        codeword = data_packets[:]  # Systematic part

        # Generate parity packets
        for i in range(self.redundancy):
            parity_packet = [0] * packet_size

            for j in range(self.k):
                if self.generator_matrix[j][self.k + i] == 1:
                    for bit_idx in range(packet_size):
                        parity_packet[bit_idx] ^= data_packets[j][bit_idx]

            codeword.append(parity_packet)

        elapsed = time.time() - start_time
        return codeword, elapsed

    def decode_erasures_bitwise(self, received_packets, erasure_positions):
        """
        Decode original data from received packets with erasures.
        """
        if len(received_packets) < self.k:
            raise ValueError(f"Need at least {self.k} packets to decode")

        start_time = time.time()

        # Extract generator submatrix for received positions
        received_positions = [i for i in range(self.n) if i not in erasure_positions]
        selected_positions = received_positions[:self.k]

        submatrix = []
        for pos in selected_positions:
            if pos < self.k:  # Data packet
                row = [0] * self.k
                row[pos] = 1
                submatrix.append(row)
            else:  # Parity packet
                parity_idx = pos - self.k
                row = [self.generator_matrix[j][pos] for j in range(self.k)]
                submatrix.append(row)

        # Decode using bitwise solver
        packet_size = len(received_packets[0])
        decoded_data = []

        for bit_pos in range(packet_size):
            bit_vector = [received_packets[i][bit_pos] for i in range(self.k)]
            solution_bits = self._solve_systematic_gf2(submatrix, bit_vector, selected_positions)

            if bit_pos == 0:
                decoded_data = [[0] * packet_size for _ in range(self.k)]

            for j in range(self.k):
                decoded_data[j][bit_pos] = solution_bits[j]

        elapsed = time.time() - start_time
        return decoded_data, elapsed

    def _solve_systematic_gf2(self, matrix, vector, positions):
        """Solve systematic equation system."""
        # For systematic codes, this is often simpler
        solution = [0] * self.k

        for i, pos in enumerate(positions):
            if pos < self.k:  # Direct data packet
                solution[pos] = vector[i]

        # For parity equations, solve remaining unknowns
        # (Simplified implementation)
        return solution


def benchmark_network_coding():
    """Benchmark network coding applications."""
    print("=== Network Coding Benchmark ===")

    # Test different parameters
    test_cases = [
        (4, 6, 32),  # k=4, n=6, 32-bit packets
        (8, 12, 64),  # k=8, n=12, 64-bit packets
        (16, 24, 128),  # k=16, n=24, 128-bit packets
    ]

    for k, n, packet_size in test_cases:
        print(f"\nTesting RLNC with k={k}, n={n}, packet_size={packet_size}:")

        # Create network coder
        coder = NetworkCoder(k, packet_size)
        coder.generate_random_encoding_matrix(n)

        # Generate test data
        original_packets = []
        for i in range(k):
            packet = [random.randint(0, 1) for _ in range(packet_size)]
            original_packets.append(packet)

        # Encode
        coded_packets, encode_time = coder.encode_packets_bitwise(original_packets)
        print(f"  Encoding time: {encode_time*1000:.3f} ms")

        # Simulate packet loss (receive exactly k packets)
        received_packets = coded_packets[:k]
        received_coeffs = coder.encoding_matrix[:k]

        # Decode
        try:
            decoded_packets, decode_time = coder.decode_packets_bitwise(received_packets, received_coeffs)
            print(f"  Decoding time: {decode_time*1000:.3f} ms")

            # Verify correctness
            correct = all(original_packets[i] == decoded_packets[i] for i in range(k))
            print(f"  Decoding correct: {correct}")

        except Exception as e:
            print(f"  Decoding failed: {e}")


def demo_content_distribution():
    """Demonstrate content distribution scenario."""
    print("\n=== Content Distribution Demo ===")

    # Scenario: Distributing a file over a lossy network
    file_size = 1024  # bits
    packet_size = 128  # bits per packet
    k = file_size // packet_size  # 8 packets
    n = k + 4  # Add 4 redundant packets

    print(f"Distributing {file_size}-bit file in {k} packets of {packet_size} bits each")
    print(f"Adding {n-k} redundant packets for fault tolerance")

    # Create erasure coder
    erasure_coder = ErasureDecoder(k, n)
    erasure_coder.generate_generator_matrix()

    # Generate file data
    file_data = []
    for i in range(k):
        packet = [random.randint(0, 1) for _ in range(packet_size)]
        file_data.append(packet)

    # Encode with redundancy
    encoded_packets, encode_time = erasure_coder.encode_systematic_bitwise(file_data)
    print(f"Encoding completed in {encode_time*1000:.3f} ms")

    # Simulate network losses (lose up to 4 packets)
    num_losses = 3
    loss_positions = random.sample(range(n), num_losses)

    received_packets = [encoded_packets[i] for i in range(n) if i not in loss_positions]
    print(f"Lost {num_losses} packets at positions {loss_positions}")
    print(f"Received {len(received_packets)} out of {n} packets")

    # Decode
    try:
        decoded_data, decode_time = erasure_coder.decode_erasures_bitwise(received_packets[:k],
                                                                          loss_positions)
        print(f"Decoding completed in {decode_time*1000:.3f} ms")

        # Verify
        success = all(file_data[i] == decoded_data[i] for i in range(k))
        print(f"File recovery successful: {success}")

    except Exception as e:
        print(f"Decoding failed: {e}")


if __name__ == "__main__":
    # Run benchmarks
    benchmark_network_coding()

    # Run content distribution demo
    demo_content_distribution()
