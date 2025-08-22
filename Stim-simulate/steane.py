import stim
import pymatching
import numpy as np

def apply_noisy_CNOT(circuit: stim.Circuit, control: list, target: list, error_prob: float):
    """
    Applies a noisy CNOT gate to the circuit with a depolarizing error model.

    Args:
        circuit: The Stim circuit to modify.
        control: The control qubit(s) for the CNOT gate.
        target: The target qubit(s) for the CNOT gate.
        error_prob: The probability of a depolarizing error after the gate.

    Returns:
        None
    """
    assert len(control) == len(target), "Control and target qubits must have the same length."
    for i in range(len(control)):
        circuit.append("CNOT", [control[i], target[i]]) # type: ignore
        # circuit.append("DEPOLARIZE2", [control[i], target[i]], error_prob)
def apply_measurement(circuit: stim.Circuit, d:list, stabs: list, ancillas: list, error_prob_set: np.ndarray):
    z_stabs, x_stabs = stabs
    az, ax = ancillas
    error_prob, error_prob_2 = error_prob_set
    for i in range(3):
        # Entangle ancilla with data qubits for the i-th Z-stabilizer
        for data_q in z_stabs[i]:
            circuit.append("CNOT", [data_q, az[i]]) # type: ignore
            circuit.append("DEPOLARIZE2", [data_q, az[i]], error_prob_2)
            
    circuit.append("MR", az) # type: ignore # Measure Z-ancillas
    # circuit.append("DEPOLARIZE1", az, error_prob)
    circuit.append("X_ERROR", az, error_prob)

    circuit.append("R", ax) # type: ignore # Reset X-ancillas
    circuit.append("DEPOLARIZE1", ax, error_prob)
    circuit.append("H", d) # type: ignore # Rotate data qubits to X-basis
    circuit.append("DEPOLARIZE1", d, error_prob)
    # circuit.append("H", ax)
    circuit.append("TICK") # type: ignore # Mark the end of a time step
    for i in range(3):
        # Entangle ancilla with data qubits for the i-th X-stabilizer
        for data_q in x_stabs[i]:
            circuit.append("CNOT", [data_q, ax[i]]) # type: ignore
            circuit.append("DEPOLARIZE2", [data_q, ax[i]], error_prob_2)
            # circuit.append("H", ax[i]) # Rotate ancillas back to Z-basis
    circuit.append("H", d) # type: ignore # type: ignore # Rotate data qubits back
    circuit.append("DEPOLARIZE1", d, error_prob)
    circuit.append("MR", ax) # type: ignore # type: ignore # Measure X-ancillas
    circuit.append("X_ERROR", ax, error_prob) 



def create_steane_circuit(rounds: int, error_prob_set: np.ndarray) -> stim.Circuit:
    """
    Creates a Stim circuit for the Steane code with repeated syndrome measurements.

    Args:
        rounds: The number of times to repeat the stabilizer measurements.
        error_prob: The probability of a depolarizing error after each gate.

    Returns:
        A stim.Circuit object.
    """
    # The Steane code is a [[7,1,3]] code.
    # Data qubits: 0-6
    # Z-stabilizer ancillas: 7-9
    # X-stabilizer ancillas: 10-12
    circuit = stim.Circuit()
    
    # Define qubit groups for clarity
    d = [0, 1, 2, 3, 4, 5, 6] # Data qubits
    az = [7, 8, 9]           # Z-ancillas
    ax = [10, 11, 12]        # X-ancillas

    # Define the stabilizer checks for the Steane code
    # K1 = IIIZZZZ, K2 = ZIZIZIZ, K3 = IZZIIZZ 
    # K4 = IIIXXXX, K5 = XIXIXIX, K6 = IXXIIXX 
    z_stabs = [[d[3], d[4], d[5], d[6]], [d[0], d[2], d[4], d[6]], [d[1], d[2], d[5], d[6]]]
    x_stabs = [[d[3], d[4], d[5], d[6]], [d[0], d[2], d[4], d[6]], [d[1], d[2], d[5], d[6]]]

    error_prob, error_prob_2 = error_prob_set
    ### Encoding (noise-free)
    circuit.append("R", az) # type: ignore # Reset Z-ancillas
    circuit.append("R", d) # type: ignore # Reset data qubits
    circuit.append("H", [d[0], d[1], d[3]]) # type: ignore
    
    controls = [d[0], d[3], d[1], d[0], d[3], d[1], d[0], d[1], d[3]]
    targets = [d[2], d[5], d[6], d[4], d[6], d[5], d[6], d[2], d[4]]
    apply_noisy_CNOT(circuit, controls, targets, 0)
    circuit.append("TICK") # type: ignore


    ### Apply a noisy logical Hadamard gate

    circuit.append("H", d) # type: ignore # Rotate data qubits to X-basis
    circuit.append("DEPOLARIZE1", d, error_prob) # type: ignore # Apply depolarizing noise to data qubits
    # error_prob_set = [0,0]
    apply_measurement(circuit, d, [z_stabs, x_stabs], [az, ax], error_prob_set)
    num_measurements_in_round = len(az) + len(ax)
    for i in range(len(az) + len(ax)):
        circuit.append("DETECTOR", [stim.target_rec(- num_measurements_in_round + i)]) # type: ignore # type: ignore
    circuit.append("TICK") # type: ignore # Mark the end of a time step
    # --- Main Loop: Repeat measurements for 'rounds' ---
    for r in range(rounds):
        # -- Z-basis stabilizer measurements --
        apply_measurement(circuit, d, [z_stabs, x_stabs], [az, ax], error_prob_set)
        # num_measurements_in_round = len(az) + len(ax)
        for i in range(num_measurements_in_round):
            circuit.append("DETECTOR", [stim.target_rec(- num_measurements_in_round - i - 1), stim.target_rec(-i - 1)]) # type: ignore # type: ignore
        circuit.append("TICK") # type: ignore # Mark the end of a time step

    ## Error-free final measurement
    for i in range(len(az)):
        for dq in z_stabs[i]:
            circuit.append("CNOT", [dq, az[i]]) # type: ignore
    circuit.append("MR", az) # type: ignore

    circuit.append("H", d) # type: ignore
    for i in range(len(ax)):
        for dq in x_stabs[i]:
            circuit.append("CNOT", [dq, ax[i]]) # type: ignore
    
    circuit.append("MR", ax) # type: ignore
    circuit.append("H", d) # type: ignore  
    
    for i in range(num_measurements_in_round):
        circuit.append("DETECTOR", [stim.target_rec(- num_measurements_in_round - i - 1), stim.target_rec(-i - 1)]) # type: ignore # type: ignore
    circuit.append("TICK") # type: ignore # Mark the end of a time step

    # --- Virtual Logical Measurement ---

    for i in range(len(az)):
        for dq in z_stabs[i]:
            circuit.append("CNOT", [dq, az[i]]) # type: ignore
    circuit.append("MR", az) # type: ignore
    circuit.append("H", d) # type: ignore
    for i in range(len(ax)):
        for dq in x_stabs[i]:
            circuit.append("CNOT", [dq, ax[i]]) # type: ignore
     
    circuit.append("MR", ax) # type: ignore
    circuit.append("H", d) # type: ignore  


    ## Measure Logical Z operator 
    al = [13]
    circuit.append("R", al) # type: ignore # Reset logical ancilla
    circuit.append("H", d) # type: ignore
    for i in range(7):
        circuit.append("CNOT", [d[i], al[0]]) # type: ignore ## (Virtual) Logical Z measurement
    circuit.append("MR", al) # type: ignore
    circuit.append("H", d) # type: ignore
     
    circuit.append("TICK") # type: ignore
    for i in range(6): 
        circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-i-2), stim.target_rec(- i - num_measurements_in_round - 2)], i + 1)
    circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], 0)
    # circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-7)], 4)
    
    # circuit.append("DETECTOR", [stim.target_rec(-1), stim.target_rec(-3), stim.target_rec(-5), stim.target_rec(-7), stim.target_rec(-5-num_measurements_in_round)])
    # circuit.append("DETECTOR", [stim.target_rec(-1), stim.target_rec(-2), stim.target_rec(-5), stim.target_rec(-6), stim.target_rec(-6-num_measurements_in_round)])
    # circuit.append("DETECTOR", [stim.target_rec(-1), stim.target_rec(-2), stim.target_rec(-3), stim.target_rec(-4), stim.target_rec(-7-num_measurements_in_round)])
    
    # The logical observable is the parity of all physical Z measurements.
    # circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1), stim.target_rec(-2), stim.target_rec(-3), stim.target_rec(-4), stim.target_rec(-5), stim.target_rec(-6), stim.target_rec(-7)], 0)

    
    return circuit

# --- Simulation and Decoding ---

# 1. Set Parameters
NUM_SHOTS = 200000
ROUNDS = 0
ERROR_PROB = [0.0003, 0.006]

# 2. Create the circuit
steane_circuit = create_steane_circuit(rounds=ROUNDS, error_prob_set=ERROR_PROB) # type: ignore

# print(repr(steane_circuit))
# exit(0)

# 3. Create the Pymatching decoder from the circuit's error model
# This analyzes the circuit to build the syndrome graph automatically.
error_model = steane_circuit.detector_error_model(decompose_errors=True)
# print(error_model)
matching = pymatching.Matching.from_detector_error_model(error_model)

# 4. Sample the circuit to get syndromes and actual logical outcomes
sampler = steane_circuit.compile_detector_sampler()
detection_events, actual_observable_flips = sampler.sample(
    shots= NUM_SHOTS,
    separate_observables=True
)
actual_logical_flips = actual_observable_flips[:, 0] # type: ignore
actual_stabilizer_flips = actual_observable_flips[:,1:] # type: ignore


# 5. Decode the syndromes with Pymatching
# predicted_observable_flips = matching.decode_batch(detection_events)
predicted_observable_flips = matching.decode_batch(detection_events) # type: ignore
# predicted_qubit_flips = matching.decode_to_edges_array(detection_events)
predicted_logical_flips = predicted_observable_flips[:, 0] # type: ignore
predicted_stabilizer_flips = predicted_observable_flips[:, 1:] # type: ignore

# 6. Analyze the results
# A logical error occurs if the decoder's prediction doesn't match the actual outcome.
# print(f"Actual logical flips: {actual_logical_flips}")
# print(f"Predicted logical flips: {predicted_logical_flips}")
# print(f"Actual stabilizer flips: {actual_stabilizer_flips}")
# print(f"Predicted stabilizer flips: {predicted_stabilizer_flips}")


# stab_indices = (actual_logical_flips == predicted_logical_flips) & (actual_stabilizer_flips != predicted_stabilizer_flips).any(axis = 1) # type: ignore
# log_indices = (actual_logical_flips != predicted_logical_flips) # type: ignore
# print(actual_stabilizer_flips[np.where(stab_indices)[0]])
# print(predicted_stabilizer_flips[[np.where(stab_indices)[0]]])

num_log_errors = np.sum((actual_logical_flips != predicted_logical_flips)) # type: ignore
num_physical_errors = np.sum((actual_logical_flips == predicted_logical_flips) & (actual_stabilizer_flips != predicted_stabilizer_flips).any(axis = 1)) # type: ignore
# num_errors = np.sum(actual_observable_flips.flatten() != predicted_observable_flips.flatten()) # type: ignore
logical_error_rate = num_log_errors / NUM_SHOTS
physical_error_rate = num_physical_errors / NUM_SHOTS

print("" + "="*50)
print(f"Simulated {NUM_SHOTS} shots of a Steane code circuit.")
print(f"Number of measurement rounds per shot: {ROUNDS}")
print(f"Physical depolarizing error rate: {ERROR_PROB}")
print("-" * 50)
print(f"Total logical errors detected: {num_log_errors}")
print(f"Logical error rate: {logical_error_rate:.6f} ({num_log_errors} / {NUM_SHOTS})")
print("-" * 50)
print(f"Total physical errors detected: {num_physical_errors}")
print(f"Physical error rate: {physical_error_rate:.6f} ({num_physical_errors} / {NUM_SHOTS})")
print("="*50)
