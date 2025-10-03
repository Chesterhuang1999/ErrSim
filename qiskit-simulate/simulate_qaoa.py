import numpy as np
import math
from qiskit import QuantumCircuit, transpile, ClassicalRegister, QuantumRegister
from qiskit.quantum_info import Statevector, Operator, Choi, DensityMatrix, partial_trace, state_fidelity
import qiskit_aer
from qiskit_aer import noise, AerSimulator
from qiskit_aer.library import save_statevector, set_statevector
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt 
from copy import deepcopy
import time

def process_qasm(file):
    with open(file, 'r') as f:
    # split() 不带参数会处理任何长度的空白
        data = [line.strip().split() for line in f]
    for line in data:
        if line[0] == 'qreg':
            n = int(line[1][2:-2])
            circuit = QuantumCircuit(n)
        elif line[0] == 'cx':
            cl, tr = line[1].strip().split(',')
            circuit.cx(int(cl), int(tr))
        elif line[0] in ['h', ]:
            qubit = int(line[1])
            circuit.h(qubit)
        elif line[0] in ['rz', 'rx']:
            angle = float(line[1][1:-2])
            qubit = int(line[2])
            circuit.rz(angle, qubit)
        
    return circuit

def process_circuit(file):
    with open(file, 'r') as f: 
        data = [line.strip().split() for line in f]
        num_qubit = sum(1 for line in data if line[0] == 'qubit')
        circuit = QuantumCircuit(num_qubit)
        register = []
        for line in data:
            if line[0] == 'qubit':
                register.append(line[1])
            elif line[0] == 'h':
                index = register.index(line[1])
                circuit.h(index)
            elif line[0] in ['Rz', 'Rx']:
                angle = float(line[1])
                index = register.index(line[2])
                if line[0] == 'Rz':
                    circuit.rz(angle, index)
                else:
                    circuit.rx(angle, index)
            elif line[0] == 'CNOT':
                control, target = line[1].split(',')
                control = register.index(control)
                target = register.index(target)
                circuit.cx(control, target)
    
    return circuit

def sim_statevector(circuit):
    noise_model = noise.NoiseModel()
    error1q = noise.pauli_error([('X', 0.0001), ('I', 0.9999)])
    error2q = noise.pauli_error([('XI', 0.0001), ('II', 0.9999)])

    noise_model.add_all_qubit_quantum_error(error1q, ['rx', 'rz', 'h'])
    noise_model.add_all_qubit_quantum_error(error2q, ['cx'])
    circuit.save_statevector() #type: ignore
    simulator = AerSimulator(method = 'statevector')
    tcirc = transpile(circuit, simulator)
    ideal_res = simulator.run(tcirc, shots = 1).result()
    ideal_sv = ideal_res.data(0)['statevector'].data

    N_SHOTS = 5000
    simulator_noisy = AerSimulator(noise_model = noise_model, method = 'statevector')
    tcirc_noisy = transpile(circuit, simulator_noisy)

    D = 0.0
    start = time.time()
    for i in range(N_SHOTS):
        
        final_res = simulator_noisy.run(tcirc_noisy, shots = 1).result().data(0)['statevector'].data
        tr_dis = math.sqrt(abs(1 - state_fidelity(ideal_sv, final_res)))
        
        D += tr_dis
    end = time.time()

    return D/N_SHOTS, end - start
def mps_from_pairs_and_bonds(sites_pairs, bonds):
    n = len(sites_pairs)
    assert len(bonds) == n - 1, "bonds 的长度应为 n-1"
    chi = [1]
    for b in bonds:
        chi.append(b.shape[0])  
    chi.append(1)
    left_fac, right_fac = [np.sqrt(np.asarray(b, dtype = np.complex128)) for b in bonds], [np.sqrt(np.asarray(b, dtype = np.complex128)) for b in bonds]

    tensors = []
    for k, (v0, v1) in enumerate(sites_pairs):
        chiL = chi[k]
        chiR = chi[k+1]
       
        v0r = np.asarray(v0).reshape(chiL, chiR)
        v1r = np.asarray(v1).reshape(chiL, chiR)
        if k > 0:
            fac = right_fac[k - 1].reshape(chiL, 1)
            v0r = fac * v0r
            v1r = fac * v1r
        if k < n - 1:
            fac = left_fac[k].reshape(1, chiR)
            v0r = v0r * fac
            v1r = v1r * fac
        Ak = np.empty((chiL, 2, chiR), dtype=np.complex128)
        Ak[:, 0, :] = v0r
        Ak[:, 1, :] = v1r
        tensors.append(Ak)
    return tensors, chi
def fidelity_MPS(state1, state2):

    tensorA = mps_from_pairs_and_bonds(state1[0], state1[1])[0]
    tensorB = mps_from_pairs_and_bonds(state2[0], state2[1])[0]


    assert len(tensorA) == len(tensorB)
    L = np.array([[1.0+0.0j]])  # 初始左环境
    for A, B in zip(tensorA, tensorB):


        chiL_A, d, chiR_A = A.shape
        chiL_B, _, chiR_B = B.shape

        newL = np.zeros((chiR_A, chiR_B), dtype=complex)
        # 对每个物理指标 i 累加
        for i in range(d):

            A_i = A[:, i, :]                 
            B_i = B[:, i, :]                 

            newL += A_i.conj().T @ L @ B_i
        L = newL
    # L 应为 1x1（两端边界维均为 1）
    assert L.shape == (1, 1)
    return float(np.abs(L[0, 0])**2)

def sim_MPS(circuit, max_bond):
    noise_model = noise.NoiseModel()
    error1q = noise.pauli_error([('X', 0.0001), ('I', 0.9999)])
    error2q = noise.pauli_error([('XI', 0.0001), ('II', 0.9999)])
    
                                #  matrix_product_state_max_bond_dimension = max_bond,
    noise_model.add_all_qubit_quantum_error(error1q, ['rx', 'rz', 'h'])
    noise_model.add_all_qubit_quantum_error(error2q, ['cx'])
    
    simulator = AerSimulator(method = 'matrix_product_state')
    tcirc = transpile(circuit, simulator)
    tcirc.save_matrix_product_state() 
    ideal_res = simulator.run(tcirc, shots = 1).result()
    ideal_mps = ideal_res.data(0)['matrix_product_state']
    simulator_noisy = AerSimulator(noise_model = noise_model, method = 'matrix_product_state')
                                #  matrix_product_state_max_bond_dimension = max_bond)
    

    tcirc_noisy = transpile(circuit, simulator_noisy)
    tcirc_noisy.save_matrix_product_state()
    N_SHOTS = 100000

    D = 0.0
    start0 = time.time()
    start = start0
    for i in range(N_SHOTS):
        
        final_res = simulator_noisy.run(tcirc_noisy, shots = 1).result().data(0)['matrix_product_state']
        tr_dis = math.sqrt(abs(1 - fidelity_MPS(ideal_mps, final_res)))
        D += tr_dis

        if i % 10000 == 0 and i > 0:
            end = time.time()
            print(f"Finished {i} shots, time used: {end - start}s")
            start = end
    end = time.time()

    return D/N_SHOTS, end - start0
if __name__ == "__main__":

    # circuit = QuantumCircuit(4)
    # circuit.h(0)
    # for i in range(0, 3):
    #     circuit.cx(i, i+1)
    #     circuit.h(i)
    # noise_model = noise.NoiseModel()
    # error1q = noise.pauli_error([('X', 0.5), ('I', 0.5)])
    # noise_model.add_all_qubit_quantum_error(error1q, ['rx', 'rz', 'h'])
    # circuit.save_matrix_product_state()
    # simulator = AerSimulator(method = 'matrix_product_state')
    # tcirc = transpile(circuit, simulator)
    # ideal_res = simulator.run(tcirc, shots = 1).result()
    # ideal_mps = ideal_res.data(0)['matrix_product_state']
    # print(ideal_mps)
    # simulator_noisy = AerSimulator(noise_model = noise_model, method = 'matrix_product_state')
    # tcirc_noisy = transpile(circuit, simulator_noisy)
    # exit(0)
    candidate_circuit = ['QAOA_line_10','QAOA4reg_20', 'QAOARandom20', 'Isingmodel10', 'QAOA4reg_30', 'QAOA50', 'Isingmodel45']
    
    for c in candidate_circuit:
        circuit = process_circuit(f"./qasm/{c}.qasm")
        
        # distance, sim_time = sim_statevector(circuit)
        distance, sim_time = sim_MPS(circuit, 128)
        print(f"======================================")
        print(f"Time for simulating circuit {c}: {sim_time}s")
        print(f"Simulated trace distance of circuit {c}: {distance}")
        print("======================================")
