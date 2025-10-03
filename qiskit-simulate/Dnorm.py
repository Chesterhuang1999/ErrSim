from qiskit.quantum_info import SuperOp, Operator, Choi, diamond_norm
import numpy as np
import cvxpy

import math
from qiskit.quantum_info import partial_trace, DensityMatrix
def cvx_bmat(r0_r, r0_i):
    """Construct a block matrix from real and imaginary parts."""
    return cvxpy.bmat([[r0_r, -r0_i], [r0_i, r0_r]])
def rho_constr_dnorm(rho, delta, choi, subarg, solver = 'SCS', **kwargs):
    """ Construct the rho-delta diamond distance of two choi operators."""
    from scipy import sparse

    dim_in = choi._input_dim
    dim_out = choi._output_dim
    size = dim_in * dim_out

    r0_r = cvxpy.Variable((dim_in, dim_in))
    r0_i = cvxpy.Variable((dim_in, dim_in))
    r0 = cvx_bmat(r0_r, r0_i)

    ##Compute the partial trace of density
    rhop = partial_trace(rho, subarg)
    rhop_r = rhop.real
    rhop_i = rhop.imag

    x_r = cvxpy.Variable((size, size))
    x_i = cvxpy.Variable((size, size))
    x = cvx_bmat(x_r, x_i)
    iden = sparse.eye(dim_in)

    ## I\otimes\rho -  W
    c_r = cvxpy.kron(iden, r0_r) - x_r
    c_i = cvxpy.kron(iden, r0_i) - x_i
    c = cvx_bmat(c_r, c_i)
    ## W <= I 
    iden_kron = sparse.eye(size)
    iden_kron_r = iden_kron.real
    iden_kron_i = iden_kron.imag
    minw_r = iden_kron_r - x_r 
    minw_i = iden_kron_i - x_i
    minw = cvx_bmat(minw_r, minw_i)
    
    choi_rt = np.transpose(
        np.reshape(choi.data, (dim_in, dim_out, dim_in, dim_out)), (3, 2, 1, 0)
    ).reshape(choi.data.shape)
    
    choi_rt_r = choi_rt.real
    choi_rt_i = choi_rt.imag
    #### Constraints
    cons = [
        r0 >> 0,
        r0_r == r0_r.T,
        r0_i == -r0_i.T,
        cvxpy.trace(r0_r) == 1,
        x >> 0, ## W >= 0
        minw >> 0,  ## W <= I
        c >> 0,
        cvxpy.trace(rhop_r @ r0_r) + cvxpy.trace(rhop_i @ r0_i) >= cvxpy.norm(rhop, 'fro')* (cvxpy.norm(rhop, 'fro') - delta),
    ]

    # Objective function
    obj = cvxpy.Maximize(cvxpy.trace(choi_rt_r @ x_r) + cvxpy.trace(choi_rt_i @ x_i))
    prob = cvxpy.Problem(obj, cons)
    sol = prob.solve(solver=solver, **kwargs)
    print("Status:", prob.status)
    print("Optimal value of rho0:")
    with np.printoptions(precision=3, suppress=True):
        print(r0_r.value + 1j * r0_i.value)
        # print("Real part:", r0_r.value)
        # print("Imaginary part:", r0_i.value)
    return 2 * sol




def Q_constr_dnorm(choi, Q, lamb, solver: str = "SCS", **kwargs) -> float:
    """Construct the Q-lambda diamond distance of two choi operators."""
    """the Input state rho is constrained by tr(Q rho) >= lambda"""
    """Equal to the convex optimization problem:"""
    """maximize tr(choi^T W)"""
    """subject to W >= 0, tr(W) = 1, rho >=0, tr(rho) = 1, I otimes rho >= W, tr(Q rho) >= lambda"""

    from scipy import sparse
    
    dim_in = choi._input_dim
    dim_out = choi._output_dim

    size = dim_in * dim_out
    # size = dim_in 
    r0_r = cvxpy.Variable((dim_in, dim_in))
    r0_i = cvxpy.Variable((dim_in, dim_in))
    r0 = cvx_bmat(r0_r, r0_i)

    ## Variable x is the objective function W
    x_r = cvxpy.Variable((size, size))
    x_i = cvxpy.Variable((size, size))
    x = cvx_bmat(x_r, x_i)
    iden = sparse.eye(dim_in)

    ## Compute I\otimes\rho - W

    c_r = cvxpy.kron(iden, r0_r) - x_r
    c_i = cvxpy.kron(iden, r0_i) - x_i
    c = cvx_bmat(c_r, c_i)
    
    Q_r = Q.real
    Q_i = Q.imag
    ## W <= I 
    iden_kron = sparse.eye(size)
    iden_kron_r = iden_kron.real
    iden_kron_i = iden_kron.imag
    minw_r = iden_kron_r - x_r 
    minw_i = iden_kron_i - x_i
    minw = cvx_bmat(minw_r, minw_i)

    
    ## Do a transpose of Choi matrix
    choi_rt = np.transpose(
        np.reshape(choi.data, (dim_in, dim_out, dim_in, dim_out)), (3, 2, 1, 0)
    ).reshape(choi.data.shape)
    
    choi_rt_r = choi_rt.real
    choi_rt_i = choi_rt.imag
    #### Constraints
    cons = [
        r0 >> 0,
        r0_r == r0_r.T,
        r0_i == -r0_i.T,
        cvxpy.trace(r0_r) == 1,
        x >> 0, ## W >= 0
        minw >> 0,  ## W <= I
        c >> 0,
        cvxpy.trace(Q_r @ r0_r) >= lamb,
    ]

    # Objective function
    obj = cvxpy.Maximize(cvxpy.trace(choi_rt_r @ x_r) + cvxpy.trace(choi_rt_i @ x_i))
    prob = cvxpy.Problem(obj, cons)
    sol = prob.solve(solver=solver, **kwargs)
    print("Status:", prob.status)
    print("Optimal value of rho0:")
    with np.printoptions(precision=3, suppress=True):
        print(r0_r.value + 1j * r0_i.value)
        # print("Real part:", r0_r.value)
        # print("Imaginary part:", r0_i.value)
    return 2 * sol


## Create a noise-free CNOT channel and a noisy version with depolarizing noise 
from qiskit.circuit.library import CXGate, HGate
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

qc = QuantumCircuit(2)
qc.append(CXGate(), [0, 1])
qc.append(HGate(), [0])
qc.append(HGate(), [1])
CXChannel = Choi(CXGate())
HChannel = Choi(HGate())

pauli_labels_2q = ["II", "IX", "IY", "IZ", 
                "XI", "XX", "XY", "XZ", 
                "YI", "YX", "YY", "YZ", 
                "ZI", "ZX", "ZY", "ZZ"]
pauli_labels_1q = ["I", "X", "Y", "Z"]
# p = 0.005
def depol_channel_2q(p, num_qubits):
    """ Create a depolarizing channel with probability p. """
    op = Choi(np.zeros((4**num_qubits, 4**num_qubits)))
    d = 4
    coeffs = [1 - p]
    for i in range(1, d**num_qubits - 1):
        coeffs.append(p / (d**num_qubits - 1))
    for coeff, label in zip(coeffs, pauli_labels_2q):
        op = op + coeff * Choi(Operator.from_label(label))
    
    return op

def depol_channel_1q(p):
    """ Create a depolarizing channel with probability p. """
    op = Choi(np.zeros((4, 4)))
    d = 4
    coeffs = [1 - p]
    for i in range(1, d - 1):
        coeffs.append(p / (d - 1))
    for coeff, label in zip(coeffs, pauli_labels_1q):
        op = op + coeff * Choi(Operator.from_label(label))
    
    return op


from qiskit.converters import circuit_to_dag
from collections import defaultdict
## Construct eigenstates as precondition
qc = QuantumCircuit(2)
qc.h(0)
qc.h(1)
qc.cx(0, 1)
def basis_proj(num_qubits):
    """Cretate the projectors for the computational basis states."""
    projs = []
    for i in range(2**num_qubits):
        proj = np.zeros((2**num_qubits, 2**num_qubits), dtype=complex)
        proj[i, i] = 1
        projs.append([proj])
    return projs

### Compute the ideal evolution of projectors
# circuit = defaultdict(list)
# circuit = {"0": [[HGate(), [0]], [ HGate(),[1]]],"1": [[CXGate(), [0, 1]]]}
circuit = [[HGate(), [0]], [HGate(), [1]], [CXGate(), [0, 1]]]
def ideal_evolution(numq, circuit):
    # output_projs = basis_proj(numq)
    Statevector.from_label('00')
    input_states = [Statevector.from_label(f"{i:0{numq}b}") for i in range(2**numq)]
    # print(input_states)
    output_projs = basis_proj(numq)
    for gateinfo in circuit:
        qc = QuantumCircuit(numq)
        qc.append(gateinfo[0], gateinfo[1])
        # qc.h(0)
        for i in range(2**numq):
            input_state = input_states[i]
            # print(input_state)
            output_state = input_state.evolve(qc)
            # print(output_state)
            input_states[i] = output_state
                   
            output_projs[i].append(output_state.to_operator().data)
    return output_projs
output_projs = ideal_evolution(2, circuit)
### Construct noisy channels for quantum gates

p2 = 0.06
p1 = 0.003
noise_channel2 = depol_channel_2q(p2, 2)
noise_channel1 = depol_channel_1q(p1)
iden = Choi(Operator.from_label("I"))
op = (1-p1) * Choi(Operator.from_label("I")) + p1 * Choi(Operator.from_label("X"))

noise_channel3 = op.tensor(iden)
cx_depo1 = noise_channel2.compose(CXChannel)
h_depo = noise_channel1.compose(HChannel)   

hdepo_tensor = h_depo.tensor(h_depo)
hchannel_tensor = HChannel.tensor(HChannel) 

total_channel_depo = cx_depo1.compose(hdepo_tensor)
total_channel = CXChannel.compose(hchannel_tensor)
# print(output_projs[0][2])
# print(output_projs[0][3])
# dnorm1 = Q_constr_dnorm(hchannel_tensor - hdepo_tensor, output_projs[3][0],1)
# dnorm2 = Q_constr_dnorm(cx_depo1 - CXChannel, output_projs[3][2], 1)
# print(dnorm1 + dnorm2)
# print(Q_constr_dnorm(total_channel_depo - total_channel, output_projs[3][0], 1) - dnorm1 - dnorm2)
# print(hchannel_tensor)


from qiskit.converters import circuit_to_dag
from collections import defaultdict
## Construct eigenstates as precondition
qc = QuantumCircuit(2)
qc.h(0)
qc.h(1)
qc.cx(0, 1)
def basis_proj(num_qubits):
    """Cretate the projectors for the computational basis states."""
    projs = []
    for i in range(2**num_qubits):
        proj = np.zeros((2**num_qubits, 2**num_qubits), dtype=complex)
        proj[i, i] = 1
        projs.append([proj])
    return projs

### Compute the ideal evolution of projectors
# circuit = defaultdict(list)
# circuit = {"0": [[HGate(), [0]], [ HGate(),[1]]],"1": [[CXGate(), [0, 1]]]}
circuit = [[HGate(), [0]], [HGate(), [1]], [CXGate(), [0, 1]]]
def ideal_evolution(numq, circuit):
    # output_projs = basis_proj(numq)
    Statevector.from_label('00')
    input_states = [Statevector.from_label(f"{i:0{numq}b}") for i in range(2**numq)]
    # print(input_states)
    output_projs = basis_proj(numq)
    for gateinfo in circuit:
        qc = QuantumCircuit(numq)
        qc.append(gateinfo[0], gateinfo[1])
        # qc.h(0)
        for i in range(2**numq):
            input_state = input_states[i]
            # print(input_state)
            output_state = input_state.evolve(qc)
            # print(output_state)
            input_states[i] = output_state
                   
            output_projs[i].append(output_state.to_operator().data)
    return output_projs
output_projs = ideal_evolution(2, circuit)
### Construct noisy channels for quantum gates

p2 = 0.06
p1 = 0.003
noise_channel2 = depol_channel_2q(p2, 2)
noise_channel1 = depol_channel_1q(p1)
iden = Choi(Operator.from_label("I"))
op = (1-p1) * Choi(Operator.from_label("I")) + p1 * Choi(Operator.from_label("X"))

noise_channel3 = op.tensor(iden)
cx_depo1 = noise_channel2.compose(CXChannel)
h_depo = noise_channel1.compose(HChannel)   

hdepo_tensor = h_depo.tensor(h_depo)
hchannel_tensor = HChannel.tensor(HChannel) 

total_channel_depo = cx_depo1.compose(hdepo_tensor)
total_channel = CXChannel.compose(hchannel_tensor)
# print(output_projs[0][2])
# print(output_projs[0][3])
# dnorm1 = Q_constr_dnorm(hchannel_tensor - hdepo_tensor, output_projs[3][0],1)
# dnorm2 = Q_constr_dnorm(cx_depo1 - CXChannel, output_projs[3][2], 1)
# print(dnorm1 + dnorm2)
# print(Q_constr_dnorm(total_channel_depo - total_channel, output_projs[3][0], 1) - dnorm1 - dnorm2)
# print(hchannel_tensor)


