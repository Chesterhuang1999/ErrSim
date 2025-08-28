# Basic memory experiment
import stim
import numpy as np

import pymatching
import matplotlib.pyplot as plt
from collections import defaultdict
from surface_code import surface_code_err_count_experiment

def generate_coords(distance):
    data_coords = []
    xstab_coords = []
    zstab_coords = []
    for j in range(distance):
        for i in range(distance):
            x, y = 2 * i + 1, 2 * j + 1
            data_coords.append(x + (2 * distance + 1 ) * (y//2))
    mid = distance // 2 + 1
    for j in range(mid):
        for i in range(distance - 1):
            x = 2 * i + 2
            if i % 2 == 0: # start with 0 in y
                y = 4 * j
                xstab_coords.append(x + (2 * distance + 1) * (y // 2))
            else: # start with 2 in y
                y = 4 * j + 2
                xstab_coords.append(x + (2 * distance + 1) * (y // 2))
    xstab_coords.sort()
    for j in range(distance - 1):
        for i in range(mid):
            y = 2 * j + 2
            if j % 2 == 0: # start with 0 in x
                x = 4 * i + 2
                zstab_coords.append(x + (2 * distance + 1) * (y // 2))
            else:
                x = 4 * i
                zstab_coords.append(x + (2 * distance + 1) * (y // 2))
    zstab_coords.sort()
    return data_coords, xstab_coords, zstab_coords

def state_preparation(distance, p, syndrome_extraction_round):
    circ = stim.Circuit.generated("surface_code:rotated_memory_x", distance=distance,
                                  rounds=3, before_round_data_depolarization=p,
                                  before_measure_flip_probability=p)

    final_measurement = circ[-distance**2 // 2 - 2:]
    
    syndrome_extraction = circ[-distance**2 // 2 - 3]
    
    circ = circ[:-distance**2 // 2 - 3]

    assert isinstance(syndrome_extraction, stim.CircuitRepeatBlock)

    if syndrome_extraction_round > 1:
        syndrome_extraction = stim.CircuitRepeatBlock(syndrome_extraction_round - 1,
                                                  syndrome_extraction.body_copy())

        circ.append(syndrome_extraction)

    return circ + final_measurement


# Modified memory experiment including two transversal Hadamard gates
# between each syndrome extraction round
def logical_H_observe(distance, p, syndrome_extraction_round, is_observe = True):
    p1, p2 = p
    circ = stim.Circuit.generated("surface_code:rotated_memory_x", distance=distance,
                                  rounds=3, after_clifford_depolarization= p2, 
                                #   before_round_data_depolarization=p,
                                  before_measure_flip_probability= p1)
    data_qubits, x_stabs, z_stabs = generate_coords(distance)
    ### Re-define the probability for 1-qubit depolarization
    
    for i, op in enumerate(circ):
        if op.name == "DEPOLARIZE1":
            op_new = stim.CircuitInstruction("DEPOLARIZE1", op.targets_copy(), [p1])
            circ_temp = circ[:i]
            circ_temp.append(op_new)
            circ_temp += circ[i + 1:]
            circ = circ_temp
    final_measurement = circ[-distance**2 // 2 - 2:]
    syndrome_extraction = circ[-distance**2 // 2 - 3]
    ## The initialization circuit
    circ = circ[:-distance**2 // 2 - 3]
    
    free_circ = stim.Circuit.generated("surface_code:rotated_memory_x", distance = distance, 
                                        rounds = 3)
    free_se = free_circ[-distance**2 // 2 - 2]
    
    assert isinstance(syndrome_extraction, stim.CircuitRepeatBlock), syndrome_extraction
    assert syndrome_extraction_round >= 1
    free_se = free_se.body_copy()
    syndrome_extraction = syndrome_extraction.body_copy()
  
    errors = stim.CircuitInstruction("DEPOLARIZE1", data_qubits, [p1])
    Hs = stim.CircuitInstruction("H", errors.targets_copy())
   
    snippet = stim.Circuit()
    snippet.append(Hs)
    snippet.append(errors)
    snippet.append(Hs)
    snippet.append(errors)
    if syndrome_extraction_round > 1:
        se = stim.CircuitRepeatBlock(syndrome_extraction_round - 1,
                                                  syndrome_extraction)
        snippet.append(se)
    else:
        snippet += syndrome_extraction
    
    
    ## Virtual (error-free) measurement to test error-rates
    
    free_stabs = free_se[:-distance**2 - 1]
    s = 2 * distance + 1
    N = 2 * distance ** 2 + distance
    free_logs_str = f"QUBIT_COORDS(0,{s - 1}) {N}\n"
    free_logs_str += f"R {N} \n TICK \n"
    free_logs_str += f"H {N}\n TICK \n"
    for i in range(distance):
        free_logs_str += f"CX {N} {i * s + 1} \n "
    free_logs_str += f"TICK \n H {N}\n TICK"
    free_logs_meas = stim.Circuit(free_logs_str)
    meas_targ = free_se[-distance**2 - 1].targets_copy()
    meas_targ.append(stim.GateTarget(N))
    meas_cmd = stim.CircuitInstruction("MR", meas_targ)
    free_meas = free_stabs + free_logs_meas
    free_meas.append(meas_cmd)
 

    ## Append to original circuit
    circ.append(snippet)

    
    ## Deal with final_measurement (replace detector with observable)
    data_meas = final_measurement[1:-distance**2 // 2]
    detector = final_measurement[-distance**2 // 2:-1]
    declare_obs = stim.Circuit("""""")
    for i, op in enumerate(detector):
        declare_obs.append("OBSERVABLE_INCLUDE", op.targets_copy()[:-1], i + 1)
        
    

    obs = final_measurement[-1]
    final_meas_new = data_meas + declare_obs
    final_meas_new.append(obs)
    
    return circ + final_meas_new
def memory_experiment_run(circuit, N):
    sampler = circuit.compile_detector_sampler()

    shots = N

    detection_events, actual_observable_flips = sampler.sample(shots, separate_observables=True)
    error_model = circuit.detector_error_model()

    matching = pymatching.Matching.from_detector_error_model(error_model)

    predicted_observable_flips = matching.decode_batch(detection_events)

    # num_log_errors_p1 = np.sum(actual_observable_flips[:,1] != predicted_observable_flips[:,1])
    num_log_errors = np.sum(actual_observable_flips[:,0] != predicted_observable_flips[:,0])

    logical_error_rate_p1 = 0
    # logical_error_rate_p1 = num_log_errors_p1 / shots
    logical_error_rate = num_log_errors / shots 

    return logical_error_rate_p1, logical_error_rate

if __name__ == "__main__":
  
    ps = [0.006 * ((0.02 / 0.006)**(i / 5)) for i in range(5)]
    ler_ft = np.zeros((4, 5))
    ler_fixed = np.zeros((4, 5))
    # for distance in range(3, 10, 2):
    #     for i, p in enumerate(ps):
    #         circuit = state_preparation_modified(distance, p, distance)
    #         ler_ft[distance // 2 - 1, i] = memory_experiment_run(circuit, 100000)
    sumofler = 0

    circ = logical_H_observe(3, (0.0003, 0.006), 1, False)

    print(circ)
    exit(0)
    for i in range(50):
        ler_p1, ler_base = memory_experiment_run(circ, 100000)
        print(ler_base)
        sumofler +=ler_base
    print( sumofler* 20 )
    
    for distance in range(3, 10, 2):
        for i, p in enumerate(ps):
            circuit = logical_H_observe(distance, p, 2)
            ler_fixed[distance // 2 - 1, i] = round(memory_experiment_run(circuit, 1000000),5)

    print(ler_fixed)
        # set logarithmic plot
    # fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # axs[0].set_title("Logical Error Rate vs Distance (Fixed)")
    # axs[0].set_xlabel("Distance")
    # axs[0].set_ylabel("Logical Error Rate")
    # axs[0].grid(True)
    # axs[0].set_yscale('log')
    # axs[0].set_xscale('linear')
    # for i in range(4):
    #     axs[0].plot(ps, ler_fixed[i], label=f"d = {2 * i + 3}")
    # axs[0].legend()

    # axs[1].set_title("Logical Error Rate vs Distance (Free)")
    # axs[1].set_xlabel("Distance")
    # axs[1].set_ylabel("Logical Error Rate")
    # axs[1].grid(True)
    # axs[1].set_yscale('log')
    # axs[1].set_xscale('linear')
    # for i in range(4):
    #     axs[1].plot(ps, ler_ft[i], label=f"d = {2 * i + 3}")
    # axs[1].legend()

    # plt.savefig("Output/Figures/H_trans_comp.png")
    plt.figure(figsize=(10, 6))
    plt.title("Transversal H + One round QEC")
    plt.xticks(range(1, 5))
    plt.grid(True)
    plt.yscale('log')
    plt.xscale('linear')
    plt.yscale("log") 


    for i in range(4):
        plt.plot(ps, ler_fixed[i], label = f"d = {2 * i + 3}")
    plt.legend()
    plt.savefig("Output/Figures/transversalH.png")