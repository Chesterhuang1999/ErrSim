import stim
import pymatching
import numpy as np
import matplotlib.pyplot as plt

import sympy
from scipy.optimize import fsolve
import numpy as np
from collections import defaultdict
from itertools import combinations

def surface_matrix_gen(n):
    H = np.zeros((n * n - 1, 2 * n * n), dtype = int)
    t = n // 2
    halfrow = (pow(n, 2) - 1) // 2
    z_cnt = 0
    x_cnt = 0
    for i in range(t):
        i2 = 2 * i
        for j in range(t):
            j2 = 2 * j
            topl_od_z = i2 * n + j2 
            topl_ev_z = (i2 + 1) * n + (j2 + 1)
            topl_od_x = (i2) * n + (j2 + 1)
            topl_ev_x = (i2 + 1) * n + (j2)
            for k in [topl_od_x, topl_ev_x]:
                H[x_cnt][k] = 1
                H[x_cnt][k + 1] = 1
                H[x_cnt][k + n] = 1
                H[x_cnt][k + n + 1] = 1
                x_cnt += 1
            for k in [topl_od_z, topl_ev_z]:
                k1 = k + n * n
                H[halfrow + z_cnt][k1] = 1
                H[halfrow + z_cnt][k1 + 1] = 1
                H[halfrow + z_cnt][k1 + n] = 1
                H[halfrow + z_cnt][k1 + n + 1] = 1
                z_cnt += 1
        
        H[x_cnt][i2] = 1
        H[x_cnt][i2 + 1] = 1
        # H[x_cnt][i2 * n] = 1
        # H[x_cnt][(i2 + 1) * n] = 1
        x_cnt += 1
        temp = i2 + 1 + (n-1) *n
        H[x_cnt][temp] = 1
        H[x_cnt][temp + 1] = 1
        # temp = (i2 + 2) * n - 1 
        # H[x_cnt][temp] = 1
        # H[x_cnt][temp + n] = 1    
        x_cnt += 1
        H[halfrow + z_cnt][(i2 + 1) * n + n * n] = 1
        H[halfrow + z_cnt][(i2 + 2) * n + n * n] = 1
        # H[halfrow + z_cnt][i2 + 1 + n * n] = 1
        # H[halfrow + z_cnt][i2 + 2 + n * n] = 1
        z_cnt += 1
        temp = (i2 + 1) * n - 1
        H[halfrow + z_cnt][temp + n * n] = 1
        H[halfrow + z_cnt][temp + (n + 1) * n] = 1
        # temp = i2 + (2*n-1) * n
        # H[halfrow + z_cnt][temp] = 1
        # H[halfrow + z_cnt][temp + 1] = 1
        z_cnt += 1
    
    HX = H[:(n * n - 1)//2, :n*n]
    HZ = H[(n * n - 1)//2:, n*n:]
    return HX, HZ 



# def generate_prob(H, w_max, p):
def generate_prob(H, w_max):
    """Generate the probability of each error pattern."""
    p = sympy.Symbol('p')
    k, n = H.shape
    H = sympy.Matrix(H)
    dist = defaultdict(float)
    for w in range(w_max + 1):
        prob_weight = (p**w) * ( (1 - p)**(n-w)) 
        # if prob_weight < 1e-10:
        #     break
        err_loc_iter = combinations(range(n), w)
        
        if w == 0:
            # snd_vec = np.zeros(k)
            snd_vec = sympy.zeros(k, 1)
            snd_tuple = tuple(int(x) for x in snd_vec)
            dist[snd_tuple] += prob_weight
            continue

        for err_loc in err_loc_iter:
            cols = [H[:, i] for i in err_loc]
            snd_vec = sympy.zeros(k ,1)
            for col in cols: 
                snd_vec += col
            snd_vec_mod2 = [elem % 2 for elem in snd_vec]
            
            # snd_vec = np.sum(H[:, err_loc], axis = 1)% 2
        
            # snd_tuple = tuple(snd_vec)

            snd_tuple = tuple(int(x) for x in snd_vec_mod2)
            dist[snd_tuple] += prob_weight
    
    return dist

def enum_syn(H, w_max):
    k, n = H.shape
    syn = []
    for w in range(w_max + 1):
        if w == 0:
            snd_vec = np.zeros(k, dtype = int)
            # snd_tuple = tuple(snd_vec)
            syn.append(snd_vec)
            continue

        err_loc_iter = combinations(range(n), w)
        for err_loc in err_loc_iter:
            snd_vec = np.sum(H[:, err_loc], axis = 1) % 2
            syn.append(snd_vec)
    
    return syn
    
    

def surface_code_circuit_transform(rounds, distance, depo_error = 0.003, flip_error = 0.001, virtual = False, phys = True):
    """Remove virtual measurements from the surface code circuit template. (guarded by virtual)"""
    """Add observables for each physical qubit (guarded by phys)"""
    circuit = stim.Circuit.generated("surface_code:rotated_memory_x", rounds = rounds, distance = distance, 
                                        after_clifford_depolarization = depo_error, after_reset_flip_probability= flip_error,
                                        before_measure_flip_probability= flip_error,
                                        )
    new_circuit = stim.Circuit()
    # Need physical errors or virtual measurement
    
    end1, end2 = 0, 0
    for index, op in enumerate(circuit):
        if op.name == "MX" and circuit[index-1].name != "MX":
            end1 = index
        if op.name != "MX" and circuit[index-1].name == "MX":
            end2 = index
            break
    # 
    new_circuit = circuit[:end1 - 1] + circuit[end1: end2] ## Remove error before virtual measurement 
    origin_circuit = circuit[:end1 - 1] + circuit[end1:]
    virtual_detector = circuit[end2:-1]
    
    if virtual == True: 
        new_circuit = new_circuit + virtual_detector

    # gap = 2 * distance + 1
    # new_circuit.append("MX", [(1+ i * gap) for i in range(distance)]) # type: ignore
    HX, HZ  = surface_matrix_gen(distance)
    new_circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-i * distance) for i in range(1, distance + 1)], 0) # type: ignore

    ### Additional stabilizer observation for physical errors
    if phys == True:
        k, n = HX.shape
        for i in range(k):
            nz = np.nonzero(HX[i])[0]
            new_circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(j - n) for j in nz], i + 1)
    # for i in range(distance ** 2):
    #     new_circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-i - 1)], i + 1) # type: ignore
    # new_circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-i) for i in range(1, distance + 1)], 0) # type: ignore
    return origin_circuit, new_circuit

def surface_code_err_count_experiment(circuit, N):
    """Count the number of physical errors in the surface code circuit."""
    sampler = circuit.compile_detector_sampler()

    shots = N

    detection_events, actual_observable_flips = sampler.sample(shots, separate_observables=True)
    error_model = circuit.detector_error_model()

    matching = pymatching.Matching.from_detector_error_model(error_model)

    predicted_observable_flips = matching.decode_batch(detection_events)

    actual_logical_flips = actual_observable_flips[:,0]
    # print(actual_logical_flips)
    actual_physical_flips = actual_observable_flips[:,1:]

    predicted_logical_flips = predicted_observable_flips[:,0] #type: ignore
    predicted_physical_flips = predicted_observable_flips[:,1:] #type: ignore

    # cond1_mask = (actual_logical_flips != predicted_logical_flips)
    # cond2_mask = np.all(actual_physical_flips == predicted_physical_flips, axis = 1)
    
    
    # num_logs_errors = np.sum(cond1_mask & cond2_mask)   
    num_logs_errors = np.sum(actual_logical_flips != predicted_logical_flips)
    
    xor_physical_flips = actual_physical_flips ^ predicted_physical_flips
    # for i in xor_physical_flips: 

    unique_rows, counts = np.unique(xor_physical_flips, axis=0, return_counts=True)
    
    
    frequency_dict = {tuple(int(x) for x in row): int(count) for row, count in zip(unique_rows, counts)}
    
    
    # num_phys_errors += np.sum(actual_physical_flips == predicted_physical_flips)

    log_error_rate = num_logs_errors / shots
    phys_err_free_rate = counts[0] / shots
    # print(f"Logical error rate: {log_error_rate} ({num_logs_errors}/{shots})")
    # phys_error_rate = num_phys_errors / shots
    # print(f"Physical error rate: {phys_error_rate} ({num_phys_errors}/{shots})")
    # return xor_physical_flips
    return log_error_rate, phys_err_free_rate, xor_physical_flips, frequency_dict



def surface_code_memory_experiment(circuit, N):
    
    # print(circuit)
    sampler = circuit.compile_detector_sampler()

    shots = N

    detection_events, actual_observable_flips = sampler.sample(shots, separate_observables=True)
    error_model = circuit.detector_error_model()

    matching = pymatching.Matching.from_detector_error_model(error_model)

    predicted_observable_flips = matching.decode_batch(detection_events)

    num_log_errors = np.sum(actual_observable_flips != predicted_observable_flips)

    logical_error_rate = num_log_errors / shots


    return logical_error_rate

def exp_value_match(f, val, init_var) -> float:
    p = sympy.Symbol('p')
    f_numeric = sympy.lambdify(p, f, modules = 'numpy')
   
    obj_fun = lambda p_val: f_numeric(p_val) - val
    p0_sol = fsolve(obj_fun, x0=init_var)
    p0 = p0_sol[0]

    return p0

def freq_fit_value(frequency_dict, err_poly, shots):
    val = next(iter(frequency_dict.values()))/shots
    init_var = 0.01
    err_free_poly = next(iter(err_poly.values()))
    p0 = exp_value_match(err_free_poly, val, init_var)
    p = sympy.Symbol('p')
    err_poly_eval = defaultdict(int)
    for key, exp in err_poly.items():
        value = exp.subs(p, p0).evalf()
        err_poly_eval[key] = round(value * shots)
    return sorted(err_poly_eval.items(), key=lambda x: x[0])

if __name__ == "__main__": 
    distance = 3
    t = (distance - 1)//2
    shots = 100000
    HX, HZ = surface_matrix_gen(distance)
    possible_syn = np.vstack(enum_syn(HX, t))
    circuit, new_circuit = surface_code_circuit_transform(1, distance, 0.003, 0.001, False, True)
    
    # print(circuit)
    err_poly = generate_prob(HX, 2)
    # print(err_free_poly)
    err_free_exp = next(iter(err_poly.values()))
    
    log_err_rate, stabs_err_free_rate, xor_physical_flips, frequency_dict = surface_code_err_count_experiment(new_circuit, shots)
    # print(xor_physical_flips)
    print(sorted(frequency_dict.items(), key = lambda x: x[0]))
    err_poly_eval = freq_fit_value(frequency_dict, err_poly, shots)
    print(err_poly_eval)
    exit(0)
    comparison_matrix = (xor_physical_flips[:, np.newaxis, :] == possible_syn)
    row_matches = np.all(comparison_matrix, axis=2)
    is_member_mask = np.any(row_matches, axis=1)
    not_in_list = np.sum(~is_member_mask)
    uncorrect_err_rate = not_in_list/shots
    print(f"Uncorrectable error: {uncorrect_err_rate + log_err_rate}")
    print(f"Calculated logical error rate:{log_err_rate}")
    # print(f"Physical err_rate:")
    # print(f"Number of unique error patterns not in the list: {not_in_list}")
    # Dist = generate_prob(HX, distance)
    # # print(Dist.get((0,0,0,0)))z
    # circuit, new_circuit = surface_code_circuit_transform(1, distance, 0.01 ,0.01, False, True)
    log_err_rate_baseline = surface_code_memory_experiment(circuit, shots)
    print(f"Baseline logical error:{log_err_rate_baseline}")
    # log_err_rate, stabs_err_free_rate = surface_code_err_count_experiment(new_circuit, shots)
    p0 = exp_value_match(err_free_poly, stabs_err_free_rate, 0.01)
    # p = sympy.Symbol('p')
    # f_numeric = sympy.lambdify(p, err_free_poly, modules='numpy')
    # obj_fun = lambda p_val: f_numeric(p_val) - stabs_err_free_rate
    # ini_guess = 0.01
    # p0_sol = fsolve(obj_fun, x0 = ini_guess)
    # p0 = p0_sol[0]
    # f_at_p0 = err_free_poly.subs(p, p0).evalf()

    phys_err_free_rate = 1 - np.power(stabs_err_free_rate, 1/(distance * distance))
    print(f"Physical err rate:{p0}")
    print(phys_err_free_rate)
    # print(log_err_rate, phys_err_free_rate)
    # # print(new_circuit)

    