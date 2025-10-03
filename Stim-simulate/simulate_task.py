from logcircuit import LogCircuit, find_max_benign_errors, postprocessing, postprocessing_CX
import stim 
import pymatching
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
def test_for_qec_block(distance, rounds, p, shots, p_prev):
    LogCir = LogCircuit(distance, rounds, p, 1)
    
    LogCir.err_free_prep()
    LogCir.insert_error(p_prev)
    LogCir.qec_block()
    LogCir.virtual_obs()
    # print(LogCir.circuit)
    error_model, log_err_rate, phys_err_rate = LogCir.run(shots, True)
    error_info = []
    for inst in error_model:
        if inst.type == "error":
            targets = [str(i) for i in inst.targets_copy()]
            error_info.append((inst.args_copy()[0],targets))
    
    ind_err = find_max_benign_errors(error_info)
    count = len(ind_err)
    values = [k for k, v in ind_err]
    # print(count, values)
    prob_dist = postprocessing(ind_err, distance // 2)
    
    return prob_dist, log_err_rate, phys_err_rate

def test_for_qec_sequential(distance, rounds, p, shots, length):
    LogCir = LogCircuit(distance, rounds, p, 2)
    LogCir.err_free_prep()
    for j in range(length):
        LogCir.log_circ_prep("CX")
        guide = "CX" if j % 2 == 0 else None
        LogCir.qec_block(guide = guide, onlyx = True)
    LogCir.virtual_obs_log(guide = "CX")
    LogCir.circuit.to_file(f"CNOT_{length}.stim")
    # print(f"Circuit Info of Length {length}: {LogCir.circuit}")
    error_model, xor_logical_flips, xor_physical_flips = LogCir.run(shots, False)
    log_err_rate = np.sum(xor_logical_flips, axis = 0) / shots
    phys_err_rate = np.sum(xor_physical_flips, axis = 0) / shots
    return log_err_rate, phys_err_rate


def test_for_hadamard(distance, rounds, p, shots):
    LogCir = LogCircuit(distance, rounds, p, 1)
    LogCir.err_free_prep()
    LogCir.log_circ_prep("H")
    LogCir.virtual_obs()
    # print(LogCir.meas_base_convert("H"))
    # exit(0)
    error_model, log_err_rate, phys_err_rate = LogCir.run(shots, True)
    error_info = []
    for inst in error_model:
        if inst.type == "error":
            targets = [str(i) for i in inst.targets_copy()]
            error_info.append((inst.args_copy()[0],targets))
    
    ind_err = find_max_benign_errors(error_info)
    count = len(ind_err)
    
    prob_dist = postprocessing(ind_err, distance // 2)
    return prob_dist, log_err_rate, phys_err_rate

def test_for_cx(distance, rounds, p, shots, p_prev = 0.0):
    LogCir = LogCircuit(distance, rounds, p, log_qubits = 2)
    LogCir.err_free_prep()
    
    LogCir.insert_error(p_prev)
    LogCir.log_circ_prep("CX")
    LogCir.virtual_obs(guide = "CX")
    print(LogCir.circuit)
    error_info = []
    error_model, log_err_rate, phys_err_rate = LogCir.run(shots, True)
    for inst in error_model:
        if inst.type == "error":
            targets = [str(i) for i in inst.targets_copy()]  #type: ignore
            error_info.append((inst.args_copy()[0],targets)) #type: ignore
    
    ind_err = find_max_benign_errors(error_info)
    count = len(ind_err)
   
    prob_dist = postprocessing_CX(ind_err, distance // 2, distance)
    return prob_dist, log_err_rate, phys_err_rate
   
def compose_prob(input, p1, p2):
    """Compute error prob distribution of composed block using total probability formula"""
    shape = input.shape
    l1 = len(p1)
    l2 = len(p2) 
    if len(shape) == 2:
        n, k = shape
        output_prob = np.zeros((n, k))
        assert n == k, "Input shape not match"
        assert l2 == k, "Input shape not match"
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                for a in range(i):
                    for b in range(j):
                        if i <= n - 1 and j <= k - 1:
                            prob1 = input[i - a - 1, j - b - 1] * p1[a] * p2[b]
                            
                            output_prob[i - 1, j - 1] += prob1
                              
                        elif i <= n - 1 and j == k:
                            output_prob[i - 1, j - 1] += input[i - a - 1, j - b - 1] * p1[a] * np.sum(p2[b:])
                        elif i == n and j <= k - 1:
                            output_prob[i - 1, j - 1] += input[i - a - 1, j - b - 1] * np.sum(p1[a:]) * p2[b]
                        else:
                            output_prob[i - 1, j - 1] += input[i - a - 1, j - b - 1] * np.sum(p1[a:]) * np.sum(p2[b:])
        return output_prob

    else: 
        n, k = shape[0], 1
        output_prob = np.zeros(n)
        assert l1 == n, "Input shape not match"
        for i in range(1, n + 1):
            for j in range(i):
                if i <= n - 1:
                    output_prob[i - 1] += input[i - j - 1] * p1[j]
                else:
                    output_prob[i - 1] += input[i - j - 1] * np.sum(p1[j:])
        assert abs(np.sum(output_prob) - 1) < 1e-8, "Output probability not normalized"
        return output_prob
    

if __name__ == "__main__":
    # Define parameters
    distance = 3
    rounds = 2
    p = (0.0001, 0.001) ## Pre-set parameters for error rate
    ler1, ler2 = [], []

    l1, l2 = test_for_qec_sequential(distance, rounds, p, shots = 1000000, length = 1)    
    print(l1, l2)
    exit(0)
    # print("======================")
    # print("======================")
    # print("======================")
    # l3, l4 = test_for_qec_sequential(distance, rounds, p, shots = 100000, length = 6)
    # exit(0)
    
    for i in range(9):
        l1_sum, l2_sum = 0, 0
        for j in range(10):
            l1, l2 = test_for_qec_sequential(distance, rounds, p, shots = 100000, length = i + 1)
            l1_sum += l1[0]
            l2_sum += l1[1]
        l1a, l2a = l1_sum / 10, l2_sum / 10

        ler1.append(l1a)
        ler2.append(l2a)
    plt.figure(figsize = (8,6))
    plt.plot(range(1, 10), ler1, marker = 'o', label = 'qubit #1 logical error rate')
    plt.plot(range(1, 10), ler2, marker = '*', label = 'qubit #2 logical error rate')
    plt.xlabel('Number of sequential CNOT gates')
    plt.ylabel('Logical error rate')
    plt.legend()
    # plt.yscale('log')
    plt.savefig('sequential_CNOT_logical_error_rate.png')
    

    # prev = 0, 0.003

    # for pr in prev:
    #     print(f"Previous error rate: {pr}")
    #     ler_sum, per_sum = 0, 0
    #     for i in range(10):
    #         prob_dist, ler, per = test_for_qec_block(distance, rounds, p, shots = 100000, p_prev = prev[1])
    #         ler_sum += ler
    #         per_sum += per
    #     ler, per = ler_sum / 10, per_sum / 10
    #     print(ler)
    #     print(per)        
    #     print(f"QEC block error distribution:{prob_dist}")
   
    # #     print('-----------------------------')
    
    # prob_dist = test_for_hadamard(distance, rounds, p, shots=100000)[0]
    # print(f"Hadamard gate error distribution:{prob_dist}")
    
    # p_prev = prob_dist[1]/(distance ** 2)
    p_prev = 8.5e-4
    prob_dist2, ler2 = test_for_cx(distance, rounds, p, shots=100000, p_prev = p_prev)[:-1]
    prob_dist1, ler1 = test_for_cx(distance, rounds, p, shots=100000, p_prev = 0)[:-1]
    print("CNOT gate error distribution:")
    print(f"CNOT err with 0 input err:\n {prob_dist1}")
    print(f"CNOT err with {p_prev} input err:\n {prob_dist2}")
    print(f"CNOT log err rate with 0 input err: {ler1}\n")
    print(f"CNOT log err rate with {p_prev} input err: {ler2}")
    p1 = np.array([0.995, 0.0049, 0.0001])
    p2 = np.array([0.998, 0.0018, 0.0002])
    # print(compose_prob(prob_dist1, p1, p2))

    # print(np.sum(prob_dist1[:-1, :-1]))