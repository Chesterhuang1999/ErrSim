import stim
import pymatching 
import numpy as np


def create_rep_circuit(distance: int, rounds: int, error_prob_set: np.ndarray) -> stim.Circuit:

    circuit = stim.Circuit("r")
    error_prob, error_prob_2 = error_prob_set
    d = [2 * i for i in range(distance)]
    a = [2 * i + 1 for i in range(distance - 1)]
    
    circuit.append("R", d) # type: ignore
    circuit.append("TICK") # type: ignore
    circuit.append("X_ERROR", d, error_prob)
    for i in range(distance - 1):
        circuit.append("CNOT", [d[i], a[i]]) # type: ignore
        circuit.append("X_ERROR", d[i], error_prob_2)
        circuit.append("X_ERROR", a[i], error_prob_2)
        circuit.append("CNOT", [d[i + 1], a[i]]) # type: ignore
        # circuit.append("DEPOLARIZE2", [d[i + 1], a[i]], error_prob_2)
        circuit.append("X_ERROR", d[i + 1], error_prob_2)
        circuit.append("X_ERROR", a[i], error_prob_2)
    circuit.append("MR", a) # type: ignore
    
    for i in range(distance - 1): 
        circuit.append("DETECTOR", [stim.target_rec(-i - 1)])         # type: ignore
    circuit.append("TICK") # type: ignore
    for j in range(rounds):
        for i in range(distance - 1):
            circuit.append("CNOT", [d[i], a[i]]) # type: ignore
            circuit.append("X_ERROR", d[i], error_prob_2)
            circuit.append("X_ERROR", a[i], error_prob_2)
            circuit.append("CNOT", [d[i + 1], a[i]]) # type: ignore
            circuit.append("X_ERROR", d[i + 1], error_prob_2)
            circuit.append("X_ERROR", a[i], error_prob_2)
        circuit.append("MR", a) # type: ignore
        for i in range(distance - 1): 
            circuit.append("DETECTOR", [stim.target_rec(-i - 1), stim.target_rec(- i - distance)]) # type: ignore


    for i in range(distance - 1):
        circuit.append("CNOT", [d[i], a[i]]) # type: ignore
        circuit.append("CNOT", [d[i + 1], a[i]]) # type: ignore
        
    circuit.append("MR", a) # type: ignore
    
    for i in range(distance - 1): 
        circuit.append("DETECTOR", [stim.target_rec(-i - 1), stim.target_rec(- i - distance)])         # type: ignore
    ### Final logical measurement

    for i in range(distance):
        circuit.append("M", d[i])
    for i in range(distance - 1):
        circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(- i - 1), stim.target_rec(- i - 2), stim.target_rec(- i - distance - 1)], i)
    circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-i - 1) for i in range(distance)], distance - 1) # type: ignore
    return circuit


if __name__ == "__main__":  
    # Set parameters
    distance = 7
    rounds = 7
    error_prob_set = np.array([0.0003, 0.001])
    NUM_SHOTS = 1000000
   
    # Create the repetition code circuit
    circuit = create_rep_circuit(distance, rounds, error_prob_set)
    
    # Print the circuit
    print(circuit)

    error_model = circuit.detector_error_model(decompose_errors=True)

    matching = pymatching.Matching.from_detector_error_model(error_model)

    # 4. Sample the circuit to get syndromes and actual logical outcomes
    sampler = circuit.compile_detector_sampler()
    detection_events, actual_observable_flips = sampler.sample(
        shots= NUM_SHOTS,
        separate_observables=True
    )
    actual_logical_flips = actual_observable_flips[:, -1] # type: ignore
    actual_stabilizer_flips = actual_observable_flips[:,:-1] # type: ignore


# 5. Decode the syndromes with Pymatching

    predicted_observable_flips = matching.decode_batch(detection_events) # type: ignore
    # predicted_qubit_flips = matching.decode_to_edges_array(detection_events)
    predicted_logical_flips = predicted_observable_flips[:, -1] # type: ignore
    predicted_stabilizer_flips = predicted_observable_flips[:, :-1] # type: ignore
    # 6. Analyze the results
    # A logical error occurs if the decoder's prediction doesn't match the actual outcome.

    num_log_errors = np.sum((actual_logical_flips != predicted_logical_flips)) # type: ignore
    num_physical_errors = np.sum((actual_logical_flips == predicted_logical_flips) & (actual_stabilizer_flips != predicted_stabilizer_flips).any(axis = 1)) # type: ignore
    # num_errors = np.sum(actual_observable_flips.flatten() != predicted_observable_flips.flatten()) # type: ignore
    logical_error_rate = num_log_errors / NUM_SHOTS
    physical_error_rate = num_physical_errors / NUM_SHOTS

    print("" + "="*50)
    print(f"Simulated {NUM_SHOTS} shots of a Steane code circuit.")
    print(f"Number of measurement rounds per shot: {rounds}")
    print(f"Physical depolarizing error rate: {error_prob_set}")
    print("-" * 50)
    print(f"Total logical errors detected: {num_log_errors}")
    print(f"Logical error rate: {logical_error_rate:.6f} ({num_log_errors} / {NUM_SHOTS})")
    print("-" * 50)
    print(f"Total physical errors detected: {num_physical_errors}")
    print(f"Physical error rate: {physical_error_rate:.6f} ({num_physical_errors} / {NUM_SHOTS})")
    print("="*50)
