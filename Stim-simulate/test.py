
import stim
from logcircuit import LogCircuit
from itertools import combinations
import copy
import numpy as np
from collections import defaultdict
import random
# Define the code distance
import stim

# 1. Define the problem from your joint distribution
p_l1 = 0.1
p_l2 = 0.05
p_l1_and_l2 = 0.02  # This is P(l1, l2)

pauli_string_l1 = "X0 Z1"
pauli_string_l2 = "Y3 Y4"


# 2. Decompose into mutually exclusive events for Stim
# Event A: l1 and l2 both happen
prob_A = p_l1_and_l2
error_A = f"{pauli_string_l1} {pauli_string_l2}"

# Event B: l1 happens, l2 does NOT
prob_B = p_l1 - p_l1_and_l2
error_B = pauli_string_l1

# Event C: l2 happens, l1 does NOT
prob_C = p_l2 - p_l1_and_l2
# error_C = pauli_string_l2
error_C = "DEPOLARIZE1 

# The "no error" case is implicit and has probability 1 - (prob_A + prob_B + prob_C)

# 3. Build the Stim circuit
# Let's assume this correlated error happens after some Hadamard gates
circuit_str = f"""
H 0 1 2 3 4

# --- Start of the correlated error model ---
# This block correctly models the P(l1, l2) joint distribution.
# Stim ensures that at most one of these events can fire.
CORRELATED_ERROR({prob_A}) {error_A}
CORRELATED_ERROR({prob_B}) {error_B}
CORRELATED_ERROR({prob_C}) {error_C}
# --- End of the correlated error model ---

M 0 1 2 3 4
"""

# 4. Create and print the Stim circuit object
circuit = stim.Circuit(circuit_str)

print("--- Probabilities for Stim ---")
print(f"P(l1 and l2): {prob_A:.3f}")
print(f"P(l1 only):    {prob_B:.3f}")
print(f"P(l2 only):    {prob_C:.3f}")
print(f"P(no error):  {1 - (prob_A + prob_B + prob_C):.3f}")


print("--- Generated Stim Circuit ---")
print(circuit)

# print(base_circuit)
# exit(0)
# color_circuit = stim.Circuit.generated("color_code:memory_xyz", distance = 5, rounds = 2)
# print(color_circuit)
# z_meas = stim.Circuit()
# data_qubits, x_stabs, z_stabs = generate_coords(distance)

# print('-----------------')
# print('------------------')
# print(code_basis_convert(base_circuit))
