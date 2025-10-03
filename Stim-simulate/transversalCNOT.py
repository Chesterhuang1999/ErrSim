import stim
import pymatching
import matplotlib.pyplot as plt
from logcircuit import LogCircuit
import numpy as np

### Implement a transversal CNOT gate and perform X-basis measurement

def qec_block_x(logCir, distance: int, rounds: int):
    dataq = logCir.data_qubits
    xstabs = logCir.x_stabs
    circ = stim.Circuit.generated("surface_code:rotated_memory_x", distance=logCir.distance, rounds = self.rounds + 1,
                                         after_clifford_depolarization=self.p2,
                                        after_reset_flip_probability=self.p1, before_measure_flip_probability= self.p1)
    for i, op in enumerate(circ):
        if op.name == "DEPOLARIZE1":
            op_new = stim.CircuitInstruction("DEPOLARIZE1", op.targets_copy(), [self.p1])
            circ_temp = circ[:i]
            circ_temp.append(op_new)
            
            circ_temp += circ[i + 1:]
            circ = circ_temp
    cut1 = - (logCir.distance ** 2 // 2) -2
    cut2 = 2 * logCir.distance ** 2 + 1
    se = circ[cut2: cut1 - 1]
    logCir.circuit += se

if __name__ == "__main__":
    # Define parameters
    distance = 3
    rounds = 2
    p = (0.0001, 0.001) ## Pre-set parameters for error rate
    ler1, ler2 = [], []
    logCir = LogCircuit(distance, rounds, p, 2)
    logCir.err_free_prep()
    logCir.log_circ_prep("CX")
    logCir.qec_block("CX", onlyx = True)
    # logCir.qec_block(guide = "CX")
    logCir.virtual_obs_log(guide = "CX")
    print(logCir.circuit)
    shots = 100000