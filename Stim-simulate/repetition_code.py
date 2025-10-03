import stim
import pymatching 
import numpy as np
from logcircuit import postprocessing

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
        circuit.append("M", d[i]) #type: ignore
    for i in range(distance - 1):
        circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(- i - 1), stim.target_rec(- i - 2), stim.target_rec(- i - distance - 1)], i)
    circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-i - 1) for i in range(distance)], distance - 1) # type: ignore
    return circuit

def find_max_benign_errors(error_info):
    symptom_winners_map = {}
    ind_error = []
    for rate, targets in error_info:
        # d_targets = tuple(sorted([t for t in targets if t.startswith("D")]))
        d_targets = tuple(sorted([t for t in targets if t.startswith("D") or t.startswith("^")]))
        if not d_targets:
            continue
        if d_targets not in symptom_winners_map or rate > symptom_winners_map[d_targets][0]:
            symptom_winners_map[tuple(d_targets)] = (rate, targets)
    for rate, targets in error_info:
        # d_targets = tuple(sorted([t for t in targets if t.startswith("D")]))
        d_targets = tuple(sorted([t for t in targets if t.startswith("D") or t.startswith("^")]))
        l_targets = tuple(sorted([t for t in targets if t.startswith("L")]))
        
        if not d_targets:
            ind_error.append((rate, targets))
            continue
        larger_rate, d_targets = symptom_winners_map[d_targets]
        if not l_targets or rate >= larger_rate:
            continue
        else:
            ## Find a malignant error, may be misunderstood
            ind_error.append((rate, targets))
    return ind_error
if __name__ == "__main__":  
    # Set parameters
    distance = 3
    rounds = 1
    error_prob_set = np.array([0.0003, 0.001])
    NUM_SHOTS = 1000000
   
    # Create the repetition code circuit
    circuit = create_rep_circuit(distance, rounds, error_prob_set)
    
    # Print the circuit
    # print(circuit)
    circuit = stim.Circuit.generated("repetition_code:memory", distance = distance, rounds = 1,
                                     after_clifford_depolarization=0.003,
                                     before_measure_flip_probability=0.001, 
                                     after_reset_flip_probability=0.001)
    
    print(circuit)
    exit(0)
    # reset = circuit[0]
    # err = circuit[1]
    # targets = [stim.GateTarget(1), stim.GateTarget(3)]
    # err = stim.CircuitInstruction("X_ERROR", targets, [0.001])
    # new_circuit = stim.Circuit()
    # new_circuit.append(reset)
    # new_circuit.append(err)
    # new_circuit += circuit[2:]
    # circuit = new_circuit
    obs = circuit[-1]
    circuit = circuit[:-distance]
    circuit.append(obs)
    for i in range(distance - 1):
        circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-i - 1), stim.target_rec(-i - 2)], i + 1)
    # circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1), stim.target_rec(-2)], 2)
    error_model = circuit.detector_error_model(decompose_errors=True)
    print(f"Circuit:{circuit}")
    print("----------------------")
    print(f"Error model:{error_model}")
    print("----------------------")
    # print(f"Error targets:")
    matching = pymatching.Matching.from_detector_error_model(error_model)
    error_info = []
    ind_err_rate = []
    for inst in error_model:
        if inst.type == "error":
            targets = [str(i) for i in inst.targets_copy()]
            error_info.append((inst.args_copy()[0],targets))

    # print(error_info)
    ind_err = find_max_benign_errors(error_info)
    print(f"Independent errors:{ind_err}")
    print(postprocessing(ind_err, 4))

    # 4. Sample the circuit to get syndromes and actual logical outcomes
    sampler = circuit.compile_detector_sampler()
    detection_events, actual_observable_flips = sampler.sample(
        shots= NUM_SHOTS,
        separate_observables=True
    )
    actual_logical_flips = actual_observable_flips[:, 0] # type: ignore
    actual_stabilizer_flips = actual_observable_flips[:,1:] # type: ignore
    



# 5. Decode the syndromes with Pymatching

    predicted_observable_flips = matching.decode_batch(detection_events) # type: ignore
    # predicted_qubit_flips = matching.decode_to_edges_array(detection_events)
    predicted_logical_flips = predicted_observable_flips[:, 0] # type: ignore
    predicted_stabilizer_flips = predicted_observable_flips[:, 1:] # type: ignore
    xor_stabilizer_flips = actual_stabilizer_flips ^ predicted_stabilizer_flips
    
    print(np.sum(np.any(xor_stabilizer_flips, axis = 1))/NUM_SHOTS)
    # 6. Analyze the results
    # A logical error occurs if the decoder's prediction doesn't match the actual outcome.

    num_log_errors = np.sum((actual_logical_flips != predicted_logical_flips)) # type: ignore
    
    num_phys_err1 = np.sum((actual_stabilizer_flips != predicted_stabilizer_flips))
    num_phys_err2 = np.sum((predicted_observable_flips[:,2]!= actual_observable_flips[:,2]))
    # num_physical_errors = np.sum((actual_logical_flips == predicted_logical_flips) & (actual_stabilizer_flips != predicted_stabilizer_flips)) # type: ignore
    # num_errors = np.sum(actual_observable_flips.flatten() != predicted_observable_flips.flatten()) # type: ignore
    logical_error_rate = num_log_errors / NUM_SHOTS
    # physical_error_rate = num_physical_errors / NUM_SHOTS
    per1, per2 = num_phys_err1 / NUM_SHOTS, num_phys_err2 / NUM_SHOTS
    print(per1, per2)
    exit(0)
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
