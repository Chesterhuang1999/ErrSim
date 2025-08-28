import stim
import numpy as np
from logical_gate import generate_coords
import pymatching
import matplotlib.pyplot as plt
from itertools import combinations
import copy
## Create a child class of stim.Circuit
class LogCircuit():
    def __init__(self, distance, rounds, p):
        self.distance = distance
        
        self.rounds = rounds
        self.p1 = p[0]
        self.p2 = p[1]
        self.data_qubits, self.x_stabs, self.z_stabs = generate_coords(self.distance)
        self.circuit = stim.Circuit()
    def err_free_prep(self):
        circuit = stim.Circuit.generated("surface_code:rotated_memory_x", distance=self.distance,
                                          rounds=self.rounds, after_clifford_depolarization=0,
                                          before_round_data_depolarization=0,
                                          before_measure_flip_probability=0)
        cut = - (self.distance **2 //2) -2
        init = circuit[:cut]
        self.circuit += init
        
    
    ## Append logical operation with noise
    def log_circ_prep(self, args):
        errors = stim.CircuitInstruction("DEPOLARIZE1", self.data_qubits, [self.p1])
        ls = stim.CircuitInstruction(args, errors.targets_copy())
        snippet = stim.Circuit()
        snippet.append(ls)
        snippet.append(errors)
        snippet.append(ls)
        snippet.append(errors)

        self.circuit += snippet
    
    ### Append virtual observations to measure error rates, with X stabs as an example
    
    def virtual_obs(self):
        circuit = stim.Circuit.generated("surface_code:rotated_memory_x", distance=self.distance,
                                          rounds=self.rounds, after_clifford_depolarization=0,
                                          before_round_data_depolarization=0,
                                          before_measure_flip_probability=0)
        cut = - (self.distance **2 //2) -2
        free_obs = circuit[cut:]
        data_obs = stim.Circuit()
        data_obs.append(free_obs[0])
        detector = free_obs[1:-1]
        for i, op in enumerate(detector):
            data_obs.append("OBSERVABLE_INCLUDE", op.targets_copy()[:-1], i + 1)
        data_obs.append(free_obs[-1])
        self.circuit += data_obs

    ### append QEC block
    def qec_block(self):
        circ = stim.Circuit.generated("surface_code:rotated_memory_x", distance=self.distance, rounds = self.rounds,
                                         after_clifford_depolarization=self.p2,
                                        after_reset_flip_probability=self.p1, before_measure_flip_probability= self.p1)
        for i, op in enumerate(circ):
            # if op.name == "DEPOLARIZE1":
            if op.name == "DEPOLARIZE1" or op.name == "DEPOLARIZE2":
                # op_new = stim.CircuitInstruction("DEPOLARIZE1", op.targets_copy(), [self.p1])
                op_new = stim.CircuitInstruction("Z_ERROR", op.targets_copy(), [self.p1])
                # op_new2 = stim.CircuitInstruction("X_ERROR", op.targets_copy(), [self.p1])
                circ_temp = circ[:i]
                circ_temp.append(op_new)
                # circ_temp.append(op_new2)
                circ_temp += circ[i + 1:]
                circ = circ_temp
        cut1 = - (self.distance **2 //2) -2
        cut2 = 2 * self.distance ** 2 + 1
        final_meas = circ[cut1:]
        # print(final_meas)
        se = circ[cut2: cut1 - 1]
        self.circuit += se
    ### simulate (silence, only error model)
    def run(self, shots):
        sampler = self.circuit.compile_detector_sampler()
        print(self.circuit)
        error_model = self.circuit.detector_error_model(decompose_errors = True)
        matching = pymatching.Matching.from_detector_error_model(error_model)
        detection_events, actual_observable_flips = sampler.sample(
            shots=shots,
            separate_observables=True
        )

        actual_logical_flips = actual_observable_flips[:, 0] # type: ignore
        actual_stabilizer_flips = actual_observable_flips[:,1:] # type: ignore
    
        predicted_observable_flips = matching.decode_batch(detection_events) # type: ignore
    
        predicted_logical_flips = predicted_observable_flips[:, 0] # type: ignore
        predicted_stabilizer_flips = predicted_observable_flips[:, 1:] # type: ignore
        xor_stabilizer_flips = actual_stabilizer_flips ^ predicted_stabilizer_flips
        num_phys_err = np.sum(np.any(xor_stabilizer_flips, axis = 1))
        num_log_errors = np.sum((predicted_logical_flips != actual_logical_flips) & (predicted_stabilizer_flips == actual_stabilizer_flips).all(axis=1))
        print(num_phys_err/shots)
        print(num_log_errors/shots)
        return error_model

    def post_process(self):
        pass

### Judging the distribution of errors using stabilizer syndromes

def postprocessing(ind_err, w_max):
    """For a set of error rate from independent sources, compute its error count prob distribution."""
    prob_dist = np.zeros((w_max + 2,))
    ind_err_rate = [rate for rate, _ in ind_err]
    n = len(ind_err_rate)
    for w in range(w_max + 1):
        prob = 1
        error_iter = combinations(range(n), w)
        if w == 0:
            for rate in ind_err_rate:
                prob *= (1 - rate)
        else:
            prob = 0
            err_rate_cpy = copy.deepcopy(ind_err_rate)
            for case in error_iter:
                prob_temp = 1
                for err in case:
                    err_rate_cpy[err] =  1 - err_rate_cpy[err]
                for rate in err_rate_cpy:
                    prob_temp *= 1 - rate
                prob += prob_temp
                err_rate_cpy = copy.deepcopy(ind_err_rate) 
        prob_dist[w] = prob
    prob_dist[-1] = 1 - np.sum(prob_dist[:-1])
    prob_dist = np.round(prob_dist, 5)
    return prob_dist

def find_max_benign_errors(error_info):
    symptom_winners_map = {}
    ind_error = []
    for rate, targets in error_info:
        d_targets = tuple(sorted([t for t in targets if t.startswith("D")]))
        # d_targets = tuple(sorted([t for t in targets if t.startswith("D") or t.startswith("^")]))
        if not d_targets:
            continue
        if d_targets not in symptom_winners_map or rate > symptom_winners_map[d_targets][0]:
            symptom_winners_map[tuple(d_targets)] = (rate, targets)
    for rate, targets in error_info:
        d_targets = tuple(sorted([t for t in targets if t.startswith("D")]))
        # d_targets = tuple(sorted([t for t in targets if t.startswith("D") or t.startswith("^")]))
        l_targets = tuple(sorted([t for t in targets if t.startswith("L")]))
        if not d_targets:
            ind_error.append((rate, targets))
            continue
        larger_rate, d_targets = symptom_winners_map[d_targets]
        if not l_targets or rate >= larger_rate:
            continue
        else:
            ## Find a malignant error, may be misunderstood
            # print(rate, targets)
            ind_error.append((rate, targets))
    return ind_error
if __name__ == "__main__":
    distance = 3
    rounds = 2
    p = (0.001, 0.003)
    shots = 100000
    arg = "H"
    LogCir = LogCircuit(distance, rounds, p)
    LogCir.err_free_prep()
    # LogCir.log_circ_prep(arg)
    LogCir.qec_block()
    LogCir.virtual_obs()
    # print(LogCir.circuit)
    error_model = LogCir.run(shots)
    # print(error_model)
    error_info = []
    for inst in error_model:
        if inst.type == "error":
            targets = [str(i) for i in inst.targets_copy()]
            error_info.append((inst.args_copy()[0],targets))
    # print(error_model)
    ind_err = find_max_benign_errors(error_info)
    count = len(ind_err)
    print(ind_err)
    # exit(0)
    print(postprocessing(ind_err, 3))
    # LogCir.qec_block()
    # print(LogCir.circuit)
