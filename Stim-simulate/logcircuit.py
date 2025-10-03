import stim
import numpy as np
from logical_gate import generate_coords
import pymatching
from beliefmatching import BeliefMatching
import matplotlib.pyplot as plt
from itertools import combinations
import copy
from surface_code import surface_matrix_gen
from collections import defaultdict
import random

# Function library
def redecompose_dem_offset(dem, coords=None, detector_offset=0):
    res = stim.DetectorErrorModel()
    initial_offset = detector_offset
    if coords is None:
        coords = dem.get_detector_coordinates()
    for inst in dem:
        if inst.type == 'error':
            targets_z = []
            targets_x = []
            for target in inst.targets_copy():
                if target.is_separator():
                    pass
                elif target.is_relative_detector_id() and sum(coords[target.val + detector_offset][:2]) % 4 == 2:
                    targets_x.append(target)
                else:
                    targets_z.append(target)
            targets = targets_z + ([stim.DemTarget.separator()] if targets_z and targets_x else []) + targets_x
            res.append(stim.DemInstruction('error', inst.args_copy(), targets))
        elif inst.type == 'repeat':
            body_new, offset = redecompose_dem_offset(inst.body_copy(), coords, detector_offset)
            res.append(stim.DemRepeatBlock(inst.repeat_count, body_new))
            detector_offset += offset * inst.repeat_count
        else:
            res.append(inst)
            if inst.type == 'shift_detectors':
                detector_offset += inst.targets_copy()[0]
    return res, detector_offset - initial_offset

def redecompose_dem(dem):
    return redecompose_dem_offset(dem)[0]
## Create a child class of stim.Circuit

class LogCircuit():
    def __init__(self, distance, rounds, p, log_qubits):
        self.distance = distance
        
        self.rounds = rounds
        self.p1 = p[0]
        self.p2 = p[1]
        self.log_qubits = log_qubits
        self.data_qubits, self.x_stabs, self.z_stabs = generate_coords(self.distance, self.log_qubits)
        
        self.circuit = stim.Circuit()
        self.template = stim.Circuit.generated("surface_code:rotated_memory_x", distance=self.distance,
                                               rounds = self.rounds, after_clifford_depolarization=self.p2, 
                                            #    after_reset_flip_probability=self.p1,
                                               before_measure_flip_probability=self.p1)
        self.matx , self.matz = surface_matrix_gen(self.distance)
    def logical_only_prep(self):
        self.circuit = self.template
    def handle_op(self, op):
        diff = 2 * self.distance** 2 + self.distance
        length = 2 * self.distance ** 2 - 1
        stabs_count_max = self.distance ** 2 - 1
        # if op.name == "MX" or op.name == "RX":
        #     return stim.CircuitInstruction(op.name, list(self.data_qubits.values())) #type: ignore
        # if op.name == "H" or op.name == "MX" or op.name == "RX":
        #     ### Expand the circuit instruction
        #     new_targ = []
        #     for i in op.targets_copy():
        #         new_targ.append(i)
        #         new_targ.append(stim.GateTarget(i.value + diff))
        #         return stim.CircuitInstruction("H", new_targ) #type: ignore
        if op.name in ["H", "MX", "RX", "R", "M", "MR", "MY", "CX"]:
            args = op.targets_copy()
            args_new = []
            args_new.extend(args)
            for j in range(1, self.log_qubits):
                args1 = [stim.GateTarget(i.value + diff * j) for i in args]
                args_new.extend(args1)
            return stim.CircuitInstruction(op.name, args_new) # type: ignore
        elif op.name in ["X_ERROR", "Y_ERROR", "Z_ERROR", "DEPOLARIZE1", "DEPOLARIZE2"]:
            args = op.targets_copy()
            rate = op.gate_args_copy()
            args_new = []
            args_new.extend(args)
            for j in range(1, self.log_qubits):
                args1 = [stim.GateTarget(i.value + diff * j) for i in args]
                args_new.extend(args1)
            return stim.CircuitInstruction(op.name, args_new, rate) # type: ignore
        elif op.name == "TICK" or op.name == "SHIFT_COORDS":
            return op
        elif op.name == "DETECTOR": ## Handling detectors
            new_circ = stim.Circuit()
            targ = op.targets_copy()
            coords = op.gate_args_copy()
            for j in range(self.log_qubits):
                trunc = self.log_qubits - j - 1
                new_coords = [coords[0], coords[1] + 2 * trunc * self.distance, coords[2]]
                if len(targ) == 1:
                    new_targ = stim.target_rec(targ[0].value - j * stabs_count_max)
                    # new_circ.append(stim.CircuitInstruction("DETECTOR", targ, new_coords))
                    new_circ.append(stim.CircuitInstruction("DETECTOR", [new_targ], new_coords))
                elif len(targ) == 2:
                    new_targ1_val = [targ[0].value - j * stabs_count_max, targ[1].value - (self.log_qubits + j - 1) * stabs_count_max]
                    # new_targ2_val = [targ[0].value - stabs_count_max, targ[1].value - self.log_qubits * stabs_count_max]
                    new_targ1 = [stim.target_rec(i) for i in new_targ1_val]
                    # new_targ2 = [stim.target_rec(i) for i in new_targ2_val]
                    new_circ.append(stim.CircuitInstruction("DETECTOR", new_targ1, new_coords))
                    # new_circ.append(stim.CircuitInstruction("DETECTOR", new_targ2, coords))
             ## Final observation (seems useless)
                else:
                    new_circ.append(op)
            return new_circ
        else:
            return op
    def err_free_prep(self):
        circuit = stim.Circuit.generated("surface_code:rotated_memory_x", distance = self.distance,
                                         rounds = 3, after_clifford_depolarization=0)
        if self.log_qubits >= 2:
            ## Cut off the final observations
            cut = - (self.distance **2 //2) -2
            circuit = circuit[:cut]
            diff = 2 * self.distance** 2 + self.distance
            length = 2 * self.distance ** 2 - 1
            # qubit_count_max = 2 * self.distance ** 2 + 3 * self.distance - 2
            new_circuit = stim.Circuit()
            new_circuit += circuit[:length]
            for i, op in enumerate(circuit):
                if i < length:
                    count = circuit[i].targets_copy()[0]
                    j, k = circuit[i].gate_args_copy()[0], circuit[i].gate_args_copy()[1]
                    for l in range(1, self.log_qubits):
                        new_targs = [int(j), int(k + 2 * l * self.distance)]
                    
                        new_circuit.append("QUBIT_COORDS",count.value + l * diff, new_targs)

               ++ 
                elif op.name == "REPEAT":
                    new_repeat_body = stim.Circuit()
                    for op_sub in op.body_copy():
                        new_repeat_body.append(self.handle_op(op_sub))
                    new_repeatblock = stim.CircuitRepeatBlock(op.repeat_count, new_repeat_body)
                    new_circuit.append(new_repeatblock)
                else:
                    new_circuit.append(self.handle_op(op))
            self.circuit += new_circuit
        else: ## test for single qubit gate
            cut = - (self.distance **2 //2) -2
            init = circuit[:cut]
            self.circuit += init
    ### Circuit preparation with noise (not memory experiment!)
    def prep(self):
        circ = self.template
        for i, op in enumerate(circ):
            if op.name == "DEPOLARIZE1":
            
                op_new = stim.CircuitInstruction("DEPOLARIZE1", op.targets_copy(), [self.p1])
                
                circ_temp = circ[:i]
                circ_temp.append(op_new)
                
                circ_temp += circ[i + 1:]
                circ = circ_temp
        cut = - (self.distance ** 2//2) - 2
        self.circuit += circ[:cut]
        # self.circuit += circ
    def insert_error(self, p):
        if isinstance(p, np.ndarray): ## Correlated error rates on each logical qubit
            n, m = p.shape
            assert n == m and n == self.log_qubits, "Input shape not match"
            qubit_count_max = self.distance ** 2
            self.circuit.append("DEPOLARIZE1", list(self.data_qubits.values())[qubit_count_max:], [p[0, 1]])
                
        else: ## Independent error rates on each qubit
            self.circuit.append("DEPOLARIZE1", list(self.data_qubits.values()), [p])

    def measurement(self):
        pass
    def meas_base_convert(self, args):
        circuit = self.template
        cut1 = 2 * self.distance**2 + 4
        init = circuit[:cut1]
  
        converted_circuit = stim.Circuit()
    
        for op in circuit[cut1:]:
            if op.name == "H":
                targets = [stim.GateTarget(i) for i in list(self.z_stabs.values())]
                converted_circuit.append("H", targets) #type: ignore
                converted_circuit.append("DEPOLARIZE1", targets, [self.p1])
            elif op.name == "TICK":
                converted_circuit.append("TICK")#type: ignore
            elif op.name == "CX":
                targ = op.targets_copy() #type: ignore
                new_targ = []
                midl = len(targ) // 2
                assert len(targ) % 2 == 0
                for i in range(midl):
                    new_targ.append(targ[2 * i + 1])
                    new_targ.append(targ[2 * i])
                converted_circuit.append("CX", new_targ) #type: ignore
                converted_circuit.append("DEPOLARIZE2", new_targ, [self.p2])
            elif op.name == "MX":
                converted_circuit.append("M", op.targets_copy()) #type: ignore
            elif op.name == "DEPOLARIZE1" or op.name == "DEPOLARIZE2":
                continue
            else:
                converted_circuit.append(op)

        return converted_circuit
    ## Append logical operation with noise
    def log_circ_prep(self, args):
        if args == "CX":
            assert self.log_qubits == 2
            # qubit_count_max = 2 * self.distance ** 2 + 3 * self.distance - 2
            diff = 2 * self.distance** 2 + self.distance    
            new_gate_targ = []
            for i, qubit in enumerate(list(self.data_qubits.values())):
                if i <= self.distance ** 2 - 1:
                    new_gate_targ.append(stim.GateTarget(qubit))
                    new_gate_targ.append(stim.GateTarget(qubit + diff))
            errors = stim.CircuitInstruction("DEPOLARIZE2", new_gate_targ,[self.p2])
            ls = stim.CircuitInstruction(args, new_gate_targ)
            snippet = stim.Circuit()
            snippet.append(ls)
            snippet.append(errors)

            self.circuit += snippet

            ### Prepare a virtual obs with reshaped stabilizer
            z_obs = stim.Circuit()

        else:
            errors = stim.CircuitInstruction("DEPOLARIZE1", list(self.data_qubits.values()), [self.p1])
            ls = stim.CircuitInstruction(args, errors.targets_copy())
            snippet = stim.Circuit()
            snippet.append(ls)
            snippet.append(errors)
            snippet.append(ls)
            snippet.append(errors)
            
            self.circuit += snippet
    ### Append virtual observations to logical error rate only
    def virtual_obs_log(self, guide = None):
        circuit = stim.Circuit.generated("surface_code:rotated_memory_x", distance = self.distance,
                                         rounds = 1, after_clifford_depolarization=0)
        cut = - (self.distance **2 //2) -2
        free_obs = circuit[cut:]
        numq = self.distance ** 2
        nums = numq - 1
        new_obs = stim.Circuit()
        for op in free_obs:
            if op.name == "DETECTOR":
                targ = op.targets_copy() #type: ignore
                coords = op.gate_args_copy() #type: ignore
                # new_obs.append(op)
                new_targ = []
                new_coords = []
                for i in range(self.log_qubits):
                    new_targ = [stim.target_rec(t.value -  numq * (self.log_qubits - i - 1)) for t in targ[:-1]]
                    t = targ[-1].value - (self.log_qubits - 1) * numq
                    new_targ.append(stim.target_rec(t  - nums * (self.log_qubits - i - 1)))
                    new_coords = [coords[0], coords[1] + 2 * self.distance * i, coords[2]]
                    new_obs.append(stim.CircuitInstruction("DETECTOR", new_targ, new_coords)) #type: ignore
            elif op.name == "MX":
                for i in range(self.log_qubits):
                    targ = [stim.GateTarget(t) for t in self.data_qubits.values()]
                new_obs.append(stim.CircuitInstruction("MX", targ))
            elif op.name == "OBSERVABLE_INCLUDE":
                targ = op.targets_copy()
                new_targ = []
            
                for i in range(self.log_qubits):
                    new_targ = [stim.target_rec(t.value - numq * (self.log_qubits - i - 1)) for t in targ]
                    new_obs.append("OBSERVABLE_INCLUDE", new_targ, i) #type: ignore

        self.circuit += new_obs

    ### Append virtual observations to measure error rates, with X stabs as an example
    def virtual_obs(self, guide = None):

        #### Reset measurement ancillae
        self.circuit.append("R", sorted(list(self.x_stabs.values()) + list(self.z_stabs.values()))) #type: ignore
        ### Z observables, via error-free measurements
        z_obs = stim.Circuit()
        z_obs.append("TICK") #type:ignore
        for coord, index in self.z_stabs.items():
            x, y = coord
            if x < 2 * self.distance:
                z_obs.append("CX", [stim.GateTarget(self.data_qubits[x + 1, y + 1]),stim.GateTarget(index)]) #type:ignore
        z_obs.append("TICK") #type:ignore
        for coord, index in self.z_stabs.items():
            x, y = coord
            if x < 2 * self.distance:
                z_obs.append("CX", [stim.GateTarget(self.data_qubits[x + 1, y - 1]),stim.GateTarget(index)]) #type:ignore
        z_obs.append("TICK") #type:ignore
        for coord, index in self.z_stabs.items():
            x, y = coord
            if x > 0:
                z_obs.append("CX", [stim.GateTarget(self.data_qubits[x - 1, y + 1]), stim.GateTarget(index)]) #type:ignore
        z_obs.append("TICK") #type:ignore
        for coord, index in self.z_stabs.items():
            x, y = coord
            if x > 0:
                z_obs.append("CX", [stim.GateTarget(self.data_qubits[x - 1, y - 1]), stim.GateTarget(index)]) #type:ignore
        z_obs.append("TICK") #type:ignore
        z_indexes = list(self.z_stabs.values())
        z_obs.append("MR", z_indexes) #type:ignore
        x_keys, z_keys = list(self.x_stabs.keys()), list(self.z_stabs.keys())
        combined_list = sorted(x_keys + z_keys, key = lambda t: (t[1], t[0]))
        z_order = []
        for key in z_keys:
            z_order.append(-combined_list.index(key) - 1)
        
        if guide == "CX":
            for i in range(len(z_indexes)):
                mid = len(z_indexes) // 2 
                if i < mid:
                    targ = [stim.target_rec(- i - 1), stim.target_rec(z_order[i]- len(z_indexes)),stim.target_rec(z_order[i + mid] - len(z_indexes))]
                else:
                    targ = [stim.target_rec(-i - 1), stim.target_rec(z_order[i] - len(z_indexes))]
                z_obs.append("OBSERVABLE_INCLUDE", targ, i + len(self.x_stabs.keys()) + self.log_qubits) #type:ignore
        else: 
            for i in range(len(z_indexes)):
                z_obs.append("OBSERVABLE_INCLUDE", [stim.target_rec(- i - 1), stim.target_rec(z_order[i] - len(z_indexes))], i + len(self.x_stabs.keys()) + self.log_qubits) #type:ignore
       
        self.circuit += z_obs

        ### X measurement, directly measure
        
        circuit = stim.Circuit.generated("surface_code:rotated_memory_x", distance=self.distance,
                                        rounds=self.rounds, after_clifford_depolarization=0,
                                        before_round_data_depolarization=0,
                                        before_measure_flip_probability=0)
        cut = - (self.distance **2 //2) -2
        
        free_obs = circuit[cut:]
        x_obs = stim.Circuit()
        data_list = list(self.data_qubits.values())

        x_obs.append("MX", data_list) #type: ignore
        qubit_count_max = len(data_list) // self.log_qubits
        detector = free_obs[1:-1]
        if guide == "CX":
            for i, op in enumerate(detector):
                
                x_obs.append("OBSERVABLE_INCLUDE", op.targets_copy()[:-1], i + 2)
                new_targ = op.targets_copy()[:-1] + [stim.target_rec(t.value - qubit_count_max) for t in op.targets_copy()[:-1]]
                
                x_obs.append("OBSERVABLE_INCLUDE", new_targ, i + 2 + len(detector) )
            
            
            new_targ = [stim.target_rec(t.value - qubit_count_max) for t in free_obs[-1].targets_copy()]
            new_targ1 = free_obs[-1].targets_copy() + new_targ
            x_obs.append("OBSERVABLE_INCLUDE", new_targ1, 0)
            x_obs.append("OBSERVABLE_INCLUDE", new_targ, 1)
            self.circuit += x_obs
        else:
            for i, op in enumerate(detector):
                
                x_obs.append("OBSERVABLE_INCLUDE", op.targets_copy()[:-1], i + self.log_qubits)
                if self.log_qubits == 2:
                    new_targ = [stim.target_rec(t.value - qubit_count_max) for t in op.targets_copy()[:-1]]
                    x_obs.append("OBSERVABLE_INCLUDE", new_targ, 2 * i + 3)
            
            x_obs.append(free_obs[-1])
            if self.log_qubits == 2:
                new_targ = [stim.target_rec(t.value - qubit_count_max) for t in free_obs[-1].targets_copy()]
                x_obs.append("OBSERVABLE_INCLUDE", new_targ, 1)
            self.circuit += x_obs

    ### append QEC block
    def qec_block(self, guide = None, onlyx = False):
        ### Generate QEC block according to the template
        
        circ = stim.Circuit.generated("surface_code:rotated_memory_x", distance=self.distance, rounds = self.rounds + 1,
                                         after_clifford_depolarization=self.p2,
                                        after_reset_flip_probability=self.p1, before_measure_flip_probability= self.p1)
        for i, op in enumerate(circ):
            if op.name == "DEPOLARIZE1":
                op_new = stim.CircuitInstruction("DEPOLARIZE1", op.targets_copy(), [self.p1])
                circ_temp = circ[:i]
                circ_temp.append(op_new)
                
                circ_temp += circ[i + 1:]
                circ = circ_temp
        cut1 = - (self.distance ** 2 // 2) -2
        cut2 = 2 * self.distance ** 2 + 1
        se = circ[cut2: cut1 - 1]
        
        z_meas = stim.Circuit()
        x_keys, z_keys = list(self.x_stabs.keys()), list(self.z_stabs.keys())
        combined_list = sorted(x_keys + z_keys, key = lambda t: (t[1], t[0]))
        z_order = {}
        for key in z_keys:
            z_order[key] = -combined_list.index(key) - 1
        
        z_index = list(self.z_stabs.values())

        if not onlyx:    
            for coords, index in self.z_stabs.items():
                if index < z_index[len(z_index) // 2]:
                    z_meas.append("DETECTOR", stim.target_rec(z_order[coords]), [coords[0], coords[1], 0])
            if guide == "CX":
                temp = se[:-1] + z_meas
                temp.append(se[-1])
                se = temp
        
        stabs_count_max = self.distance ** 2 - 1
        
        for i, op in enumerate(se): ## Memory X experiment, only X stabilizers in the state, we need to include Z stabilizers
            if op.name == "DETECTOR" and len(op.targets_copy()) < 2: #type: ignore
                new_targ = [op.targets_copy()[0], stim.target_rec(op.targets_copy()[0].value - stabs_count_max)] #type:ignore
                new_circ = se[:i]
                new_circ.append(stim.CircuitInstruction("DETECTOR", new_targ, op.gate_args_copy())) # type: ignore
                new_circ += se[i + 1:]
                se = new_circ
        new_se = stim.Circuit()
        new_se.append(stim.CircuitInstruction("SHIFT_COORDS", [], [0,0,1])) # type: ignore
        is_shift = 0
        for op in se:
            if op.name == "REPEAT":
                new_repeat_body = stim.Circuit()
                for op_sub in op.body_copy():
                    if op_sub.name == "DETECTOR":
                        t0 = op_sub.targets_copy()[0].value
                        if not onlyx or (onlyx and t0 not in list(z_order.values())):
                            
                            new_repeat_body.append(self.handle_op(op_sub))
                    else:
                        new_repeat_body.append(self.handle_op(op_sub))
                new_repeatblock = stim.CircuitRepeatBlock(op.repeat_count, new_repeat_body) #type: ignore
                new_se.append(new_repeatblock)
            
            elif op.name == "TICK":
                new_se.append(op)
            elif op.name == "SHIFT_COORDS":
                new_se.append(op)
                is_shift = 1
            else:
                new_circ = self.handle_op(op)
                
                if op.name == "DETECTOR" and guide == "CX" and is_shift == 0:
                    for i, op_new in enumerate(new_circ):
                        t0, t1 = op_new.targets_copy()
                        args = op_new.gate_args_copy()
                        if t0.value <  - stabs_count_max and t0.value not in list(z_order.values()): ## Qubit#1
                            t2 = stim.target_rec(t1.value + stabs_count_max)
                            new_se.append(stim.CircuitInstruction(op_new.name, [t0, t1, t2], op_new.gate_args_copy())) #type: ignore
                        elif t0.value >= - stabs_count_max and t0.value in list(z_order.values()): ## Qubit#2
                            t2 = stim.target_rec(t1.value - stabs_count_max)
                            new_se.append(stim.CircuitInstruction(op_new.name, [t0, t1, t2], op_new.gate_args_copy())) #type: ignore
                        else:
                            new_se.append(op_new)
                        
                else:
                    new_se.append(new_circ)
        
        self.circuit += new_se
    

    ### simulate (silence, only error model)
    def run(self, shots, open_boundary = True):
        # open_boundary = True
        sampler = self.circuit.compile_detector_sampler()
        
        error_model = self.circuit.detector_error_model(decompose_errors=True)
        # error_model = redecompose_dem(error_model)
        matching = pymatching.Matching.from_detector_error_model(error_model)
        detection_events, actual_observable_flips = sampler.sample(
            shots=shots,
            separate_observables=True
        )

        # BM = BeliefMatching(self.circuit)
        # predicted_observable_flips = BM.decode_batch(detection_events)

        actual_logical_flips = actual_observable_flips[:, :self.log_qubits] # type: ignore
        actual_stabilizer_flips = actual_observable_flips[:,self.log_qubits:] # type: ignore

        predicted_observable_flips = matching.decode_batch(detection_events) # type: ignore
        predicted_logical_flips = predicted_observable_flips[:, :self.log_qubits] # type: ignore
        xor_logical_flips = actual_logical_flips ^ predicted_logical_flips

        predicted_stabilizer_flips = predicted_observable_flips[:, self.log_qubits:] # type: ignore
        xor_stabilizer_flips = actual_stabilizer_flips ^ predicted_stabilizer_flips
        if open_boundary == True:
            actual_x_flips = actual_observable_flips[:,:self.log_qubits * (1 + (self.distance**2 // 2))]
            predicted_x_flips = predicted_observable_flips[:,:self.log_qubits * (1 + (self.distance**2 // 2))]
            xor_x_flips = actual_x_flips ^ predicted_x_flips
            log_indexes = [self.distance * i for i in range(self.distance)]
            pattern1 = []
            pattern2 = []
        
            for ind in log_indexes:
                I = []
                length = self.distance ** 2 // 2 + 1
                for i in range(length - 1):
                    targs = [j.value for j in self.circuit[-length + i].targets_copy()]
                    x = 1 if ind - self.distance ** 2 in targs else 0
                    I.append(x)
                patt1 = [1] + I
                patt2 = [0] + I
                
                pattern1.append(patt1)
                pattern2.append(patt2)
            

            # matches1 = [(xor_x_flips == p).all(axis=1) for p in pattern1]
            # matches2 = [(xor_x_flips == p).all(axis=1) for p in pattern2]

            # com_match1 = np.logical_or.reduce(matches1)
            # com_match2 = np.logical_or.reduce(matches2)
        
            # num_phys_err = np.sum(np.any(xor_stabilizer_flips, axis = 1))
            # print(f"Raw logical operator flips:{np.sum(xor_logical_flips)}")
            # print(f"=====================")
            # print(f"No logical error, flip caused by data:{np.sum(com_match1)}")

            # print(f"==================")
            # print(f"logical + data error:{np.sum(com_match2)}")
            num_phys_err = np.sum(np.any(xor_stabilizer_flips, axis = 1))
            num_log_err = np.sum(xor_logical_flips, axis = 0)
            # num_log_err = np.sum(xor_logical_flips) - np.sum(com_match1) + np.sum(com_match2)
        # print(f"Physical error rate: {num_phys_err/shots}({num_phys_err}/{shots})")
        # print(f"Logical error rate: {num_log_err/shots}({num_log_err}/{shots})")
            return error_model, num_log_err/shots, num_phys_err/shots
        else:
            return error_model, xor_logical_flips, xor_stabilizer_flips

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

def postprocessing_CX(ind_err, w_max, distance):
    """For a set of errors after CX gate, compute the joint error distribution for both qubits."""
    stab_count = (distance ** 2 - 1) // 2
    prob_dist = np.zeros((w_max + 2, w_max + 2))
    ind_err_rate = [rate for rate, _ in ind_err] 
    n = len(ind_err_rate)
    err_effect = defaultdict(list)
    i = 0
    for errs in ind_err:
        prob, source = errs
        l_targets = sorted([t for t in source if t.startswith("L")])
        object_set = {int(obs[1]) for obs in l_targets}
        a, b = 0, 0
        has_0 = 0 in object_set
        has_1 = 1 in object_set
        if has_1:
            a = 1 
        if has_0 != has_1:
            b = 1  
        if has_1 and not has_0:
            pass
        else:
            A1, A2 = np.matrix([[1, 1], [1,0]]), np.matrix([[0,1],[1,1]])
            for j in range(stab_count):
                has_c1 = (2 + j) in object_set
                has_c2 = (2 + stab_count + j) in object_set
                has_c3 = (2 + 2 * stab_count + j) in object_set
                has_c4 = (2 + 3 * stab_count + j) in object_set
                v1, v2 = np.array([has_c1, has_c2]).T, np.array([has_c3, has_c4]).T
                res1, res2 = (A1 @ v1 % 2), (A2 @ v2 % 2)
                
                a = max(res1[0,0], res2[0,0], a)
                b = max(res1[0,1], res2[0,1], b)          
    
        err_effect[i] = [prob, (a, b)]
        i = i + 1
    assert i == n

    num_trials = 100000
    outcome_counts = defaultdict(int)
    
    n = len(err_effect)
    
    # The clamping value for the state.
    clamp_val = w_max + 1
    for _ in range(num_trials): 
       
        current_x, current_y = 0, 0
       
        for i in range(n):
            prob, (dx, dy) = err_effect[i]
                   
            if random.random() < prob:
                
                current_x += dx
                current_y += dy

        final_x = min(current_x, clamp_val)
        final_y = min(current_y, clamp_val)
        
        outcome_counts[(final_x, final_y)] += 1

    prob_dist = np.zeros((w_max + 2, w_max + 2))
    for (x, y), count in outcome_counts.items():
        prob_dist[x, y] = count / num_trials


    return prob_dist

def find_max_benign_errors(error_info):
    symptom_winners_map = {}
    ind_error = []
    for rate, targets in error_info:
        d_targets = tuple(sorted([t for t in targets if t.startswith("D")]))
        
        if not d_targets:
            continue
        if d_targets not in symptom_winners_map or rate > symptom_winners_map[d_targets][0]:
            symptom_winners_map[tuple(d_targets)] = (rate, targets)
    for rate, targets in error_info:
        d_targets = tuple(sorted([t for t in targets if t.startswith("D")]))
        
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
    distance = 3
    rounds = 2
    p = (0.0001, 0.001)
    shots = 100000
    arg = "CX"

    ## Declare Circuit Instructions

    ## Test for QEC Block
    LogCir = LogCircuit(distance, rounds, p, 2)
    # LogCir.log_circ_prep("CX")
    # LogCir.qec_block(guide = "CX")
    LogCir.virtual_obs_log(guide = "CX")
    print(f"Circuit Info: {LogCir.circuit}")
    exit(0)
    ## Test for CNOT functions
    print("Experiment when d = 5")
    for p1 in range(1, 6):
        for p2 in range(1, 6):
            p = (0.0001 * p1, 0.001 * p2)
            print(f"Physical error rates: {p}")
            LogCir = LogCircuit(distance, rounds, p, 2)
            # LogCir.logical_only_prep()
            # LogCir.prep()
            LogCir.err_free_prep()
            
            # LogCir.insert_error()
            LogCir.log_circ_prep(arg)
            # LogCir.qec_block()
            
            # LogCir.qec_block()
            LogCir.virtual_obs(guide = "CX")

            # print(f"Circuit Info: {LogCir.circuit}")

            error_model, ler, per = LogCir.run(shots)
            
           
            error_info = []
            for inst in error_model:
                if inst.type == "error":
                    targets = [str(i) for i in inst.targets_copy()]
                    error_info.append((inst.args_copy()[0],targets))
           
            ind_err = find_max_benign_errors(error_info)
            count = len(ind_err)
            print(ind_err)
            prob_dist = postprocessing_CX(ind_err, distance // 2, distance)
            print(f"Joint probability distribution of error:\n{prob_dist}")
            print(f"Suggested physical error rate:{prob_dist[1,0] + prob_dist[0,1] + prob_dist[1,1]}")
            exit(0)
            # print(f"Probability distribution of error:{postprocessing(ind_err, distance // 2 + 1)}")
            print("=========================================")
    
   
