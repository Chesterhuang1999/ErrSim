import numpy as np
from qiskit import QuantumCircuit, transpile, ClassicalRegister, QuantumRegister
from qiskit.quantum_info import Statevector, Operator, Choi, DensityMatrix, partial_trace, state_fidelity
import qiskit_aer
from qiskit_aer import noise, AerSimulator
from qiskit_aer.library import save_statevector, set_statevector
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt 
from copy import deepcopy

def create_steane_code_circuit():
    # Create a quantum circuit for the Steane code
    c = ClassicalRegister(6, 'c')
    q = QuantumRegister(7, 'q')

    qc = QuantumCircuit(q, c)
    # Encoding the logical |0> state
    qc.reset(q)
    
    # Step 1: Apply Hadamard gates to qubits 0, 1, 2
    qc.h(q[0])
    qc.h(q[1])
    qc.h(q[3])

    # Step 2: Apply CNOTs according to the generator matrix
    # First layer of CNOTs
    qc.cx(q[0], q[2])
    qc.cx(q[3], q[5])
    qc.cx(q[1], q[6])

    # Second layer of CNOTs
    qc.cx(q[0], q[4])
    qc.cx(q[3], q[6])
    qc.cx(q[1], q[5])

    # Qubit 2 controls: 4, 6
    qc.cx(q[0], q[6])
    qc.cx(q[1], q[2])
    qc.cx(q[3], q[4])


    
    return qc
    
def steane_syndrome_measurement():
    # Add syndrome measurement for the Steane code
    q = QuantumRegister(7, 'q')
    c = ClassicalRegister(6, 'c')
    a = QuantumRegister(6, 'a') ## Measure qubits 

    qc_meas = QuantumCircuit(q, a, c)

    # Insert errors 
    ## Insert errors (optional, see the logical error)
    qc_meas.x(q[0])
    qc_meas.x(q[1])

    
    # Measuring Z stabilizers 
    qc_meas.cx(0, a[0])
    qc_meas.cx(2, a[0])
    qc_meas.cx(4, a[0])
    qc_meas.cx(6, a[0])

    qc_meas.cx(1, a[1])
    qc_meas.cx(2, a[1])
    qc_meas.cx(5, a[1])
    qc_meas.cx(6, a[1])

    qc_meas.cx(3, a[2])
    qc_meas.cx(4, a[2])
    qc_meas.cx(5, a[2])
    qc_meas.cx(6, a[2])

    qc_meas.barrier()
    ## Measuring X stabilizers
    qc_meas.h([0, 2, 4, 6, a[3]])
    qc_meas.cx(0, a[3])
    qc_meas.cx(2, a[3])
    qc_meas.cx(4, a[3])
    qc_meas.cx(6, a[3])
    qc_meas.h([0, 2, 4, 6, a[3]])

    qc_meas.h([1, 2, 5, 6, a[4]])
    qc_meas.cx(1, a[4])
    qc_meas.cx(2, a[4])
    qc_meas.cx(5, a[4])
    qc_meas.cx(6, a[4])
    qc_meas.h([1, 2, 5, 6, a[4]])

    qc_meas.h([3, 4, 5, 6, a[5]])
    qc_meas.cx(3, a[5])
    qc_meas.cx(4, a[5])
    qc_meas.cx(5, a[5])
    qc_meas.cx(6, a[5])
    qc_meas.h([3, 4, 5, 6, a[5]])
     
    # Measuring the logical Z operator
    
    qc_meas.barrier()
    qc_meas.measure(a, c)

    return qc_meas

def steane_decoder(syndrome : str):
    # Decode the syndrome to correct errors
    # For Steane code we have explicit expression for decoder

    #Divide and Convert str to binary number
    z_syn = syndrome[:3]

    
    x_syn = syndrome[3:]
    
    x_site = max(int(x_syn, 2) - 1, 0)
    z_site = max(int(z_syn, 2) - 1, 0)
    if x_site == 0:
        if z_site == 0:
            return None
        else:
            recoveryop = [('Z', z_site)]
    else:
        if x_site == z_site: 
            recoveryop = [('Y', x_site)]
        else:
            recoveryop = [('X', x_site)]
            if z_site != 0:
                recoveryop.append(('Z', z_site))

    return recoveryop

def steane_recovery_op(syndrome, ini_state: Statevector):
    #
    q = QuantumRegister(7, 'q') 

    a = QuantumRegister(6, 'a')
    c = ClassicalRegister(6, 'c')

    qc_recovery = QuantumCircuit(q, a, c)
    qc_recovery.set_statevector(ini_state) # type: ignore
    recovery_ops = steane_decoder(syndrome)
    
    if recovery_ops is None:
        return qc_recovery
    for op, qubit in recovery_ops:
        if op == 'X':
            qc_recovery.x(qubit)
        elif op == 'Y':
            qc_recovery.y(qubit)
        elif op == 'Z':
            qc_recovery.z(qubit)
    
    return qc_recovery

def roundsv(sv_data: np.ndarray, decimal = 4, thres = 1e-9):
    data = sv_data.copy()
    data[np.abs(data) < thres] = 0.0
    data = np.round(data, decimal)
    return Statevector(data)

def print_nonzero_val(sv_data: np.ndarray, decimal = 4):
    data = sv_data.copy()
    data_approx = roundsv(data, decimal)
    data_nonzero = np.nonzero(data_approx.data)
    data_values = data_approx.data[data_nonzero]
    return data_nonzero, data_values   
if __name__ == "__main__":
    ## Set noise strength
    P_ERR_1 = 0.0003
    P_ERR_2 = 0.006
    N_SHOTS = 2
    # Create the Steane code circuit
    dataq = QuantumRegister(7, 'q')
    ancillaq = QuantumRegister(6, 'a')
    c = ClassicalRegister(6, 'c')

    steane_detect = QuantumCircuit(dataq, ancillaq, c)
    steane_enc = create_steane_code_circuit()
    steane_meas = steane_syndrome_measurement()
    steane_detect.compose(steane_enc, inplace = True)
    steane_detect.barrier()
    steane_detect.compose(steane_meas, inplace = True)
    
    
    ideal_0L_sv = Statevector(steane_enc)
    logical_x_op = Operator.from_label("XXXXXXX")
    ideal_1L_sv = ideal_0L_sv.evolve(logical_x_op)
    
    logical_error_count = 0
    total_fidelity = 0.0 
    
    ## Set noise model
    noise_model = noise.NoiseModel()
    error_1q = noise.depolarizing_error(P_ERR_1, 1)
    error_2q = noise.depolarizing_error(P_ERR_2, 2)
    
    noise_model.add_all_qubit_quantum_error(error_1q, ['x','y','z','h','s','rz','rx', 'ry'])
    noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])
    # simulator = AerSimulator(noise_model = noise_model)
    simulator = AerSimulator()
    
    for i in range(N_SHOTS):
         
        steane_detect_copy = deepcopy(steane_detect)
        steane_detect_copy.save_statevector() # type: ignore
        
        result = simulator.run(steane_detect_copy, shots = 1).result()
        meas_sv = result.data(0)['statevector']
        rhom = partial_trace(DensityMatrix(meas_sv), [7, 8, 9, 10, 11, 12])
        print(round(state_fidelity(ideal_0L_sv, rhom), 3))
        counts = result.get_counts()
        print(counts)
        
        meas_nz, meas_nz_val = print_nonzero_val(rhom.to_statevector().data, 3)
        print(f"values in meas state: {meas_nz}, {meas_nz_val}")
        recovery = steane_recovery_op(list(counts.keys())[0], meas_sv)
        recovery.save_statevector() # type: ignore
        if recovery.depth() >= 1:
            recovery.draw('mpl', filename = f'recovery_{i+1}.png')
        
        final_sv = simulator.run(recovery, shots = 1).result().data(0)['statevector'].data
        
        
        rhof = partial_trace(DensityMatrix(final_sv), [7, 8, 9, 10, 11, 12])
        final_sv = rhof.to_statevector()
        final_nz, final_nz_val = print_nonzero_val(final_sv.data, 3)
        print(f"values in final state: {final_nz}, {final_nz_val}")
        
        fid0 = round(state_fidelity(ideal_0L_sv, rhof), 3)
        fid1 = round(state_fidelity(ideal_1L_sv, rhof), 3)
       
        print(fid0, fid1)
        total_fidelity += fid0
        if fid0 < 0.99:
            if fid1 > 0.99:
                logical_error_count += 1
                print(f"Logical error detected in shot {i+1}")
            else:
                print(f"Physical error detected in shot {i+1}")
    print(f"Average fidelity: {total_fidelity / N_SHOTS:.3f}")
    