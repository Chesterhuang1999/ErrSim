import numpy as np

# Import Qiskit

from simulate_qaoa import process_circuit
from qiskit import transpile
from qiskit_aer import AerSimulator, noise
from qiskit.quantum_info import state_fidelity
import numpy as np
import time



circuit = process_circuit("./qasm/QAOA_line_10.qasm")
        
noise_model = noise.NoiseModel()
error1q = noise.pauli_error([('X', 0.0001), ('I', 0.9999)])
error2q = noise.pauli_error([('XI', 0.0001), ('II', 0.9999)])

noise_model.add_all_qubit_quantum_error(error1q, ['rx', 'rz', 'h'])
noise_model.add_all_qubit_quantum_error(error2q, ['cx'])
# 1) 先拿理想态
sim_ideal = AerSimulator(method='statevector')
tcirc_ideal = transpile(circuit, sim_ideal)
tcirc_ideal.save_statevector()  # type: ignore
ideal_sv = sim_ideal.run(tcirc_ideal).result().data(0)['statevector']

# 2) 噪声：优先考虑密度矩阵后端（一次即可）
sim_noisy = AerSimulator(noise_model=noise_model, method='density_matrix')
tcirc_noisy = transpile(circuit, sim_noisy)
tcirc_noisy.save_density_matrix()  # type: ignore
start = time.time()
res = sim_noisy.run(tcirc_noisy).result()
rho_noisy = res.data(0)['density_matrix']
fid = state_fidelity(ideal_sv, rho_noisy)
# 若需要 trace distance 上下界
D_upper = np.sqrt(max(0.0, 1 - fid))  # 上界
D_lower = max(0.0, 1 - np.sqrt(fid))  # 下界
elapsed = time.time() - start
print(f"Fidelity={fid:.6f}, D_tr in [{D_lower:.6f}, {D_upper:.6f}], time={elapsed:.2f}s")
