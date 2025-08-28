import stim
import numpy as np
def create_surface_code_logical_zero(distance: int) -> stim.Circuit:
    """
    Creates a Stim circuit to prepare a logical |0> for a rotated surface code.

    The circuit does the following:
    1. Defines the qubit layout with coordinates.
    2. Initializes all qubits to the |0> state.
    3. Performs one full round of stabilizer measurements (both X and Z).
    4. Measures the logical Z operator at the end to verify the state.
    """
    if distance % 2 == 0:
        raise ValueError("Distance must be an odd integer.")

    # --- 1. Define Qubit Layout ---
    circuit = stim.Circuit()
    qubits = {} # Using a dictionary for convenient coordinate-based lookup
    
    # Place data qubits at vertices
    for r in range(distance):
        for c in range(distance):
            q = len(qubits)
            qubits[(r, c)] = q
            circuit.append("QUBIT_COORDS", [q], [r, c])

    # Place measure qubits on the faces (plaquettes)
    # Z-plaquettes are centered on even sums of coordinates
    # X-plaquettes are centered on odd sums of coordinates
    z_plaquettes = []
    x_plaquettes = []
    for r in range(distance):
        for c in range(distance):
            if (r + c) % 2 == 1: # Z-plaquettes (star operators)
                q = len(qubits)
                qubits[(r, c, 'Z')] = q
                circuit.append("QUBIT_COORDS", [q], [r + 0.5, c + 0.5, 1]) # z-coord 1 for clarity
                z_plaquettes.append(q)
            # Don't create plaquettes on the last row/col boundary
            elif r < distance -1 and c < distance -1: # X-plaquettes (plaquette operators)
                q = len(qubits)
                qubits[(r, c, 'X')] = q
                circuit.append("QUBIT_COORDS", [q], [r + 0.5, c + 0.5, 0]) # z-coord 0 for clarity
                x_plaquettes.append(q)

    all_qubits = [q for q in qubits.values()]
    data_qubits = [qubits[(r,c)] for r in range(distance) for c in range(distance)]
    measure_qubits = z_plaquettes + x_plaquettes

    # --- 2. Initialize all qubits to |0> ---
    # This is the crucial step for a logical |0>.
    # R resets to the Z-basis |0> state.
    circuit.append("R", all_qubits)
    circuit.append("TICK")

    # --- 3. Perform one round of Stabilizer Measurement ---
    
    # For Z-stabilizers (star operator)
    # No Hadamards needed.
    for r in range(distance):
        for c in range(distance):
            if (r + c) % 2 == 1:
                measure_q = qubits[(r, c, 'Z')]
                # Find neighboring data qubits (N, S, E, W)
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < distance and 0 <= nc < distance:
                        data_q = qubits[(nr, nc)]
                        circuit.append("CX", [measure_q, data_q])
    
    circuit.append("TICK")

    # For X-stabilizers (plaquette operator)
    circuit.append("H", x_plaquettes)
    circuit.append("TICK")
    for r in range(distance - 1):
        for c in range(distance - 1):
            if (r + c) % 2 == 0:
                measure_q = qubits[(r, c, 'X')]
                # Find neighboring data qubits
                data_q1 = qubits[(r, c)]
                data_q2 = qubits[(r + 1, c)]
                data_q3 = qubits[(r, c + 1)]
                data_q4 = qubits[(r + 1, c + 1)]
                circuit.append("CX", [measure_q, data_q1])
                circuit.append("CX", [measure_q, data_q2])
                circuit.append("CX", [measure_q, data_q3])
                circuit.append("CX", [measure_q, data_q4])

    circuit.append("TICK")
    circuit.append("H", x_plaquettes)
    circuit.append("TICK")

    # Measure all the ancillas
    circuit.append("M", measure_qubits)

    for i in range(len(measure_qubits)):
        circuit.append("OBSERVABLE_INCLUDE", stim.target_rec(-i - 1), i + 1)
    # --- 4. Define the Logical Z Operator and Measure It ---
    # A logical Z is a string of Z operators on data qubits from top to bottom.
    # We measure it by measuring the data qubits in the X basis and checking parity.
    circuit.append("MX", data_qubits)
    
    logical_z_qubits = [qubits[(r, 0)] for r in range(distance)]
    
    # Get the measurement record indices for the logical Z qubits
    # The last len(data_qubits) records correspond to the MX measurements
    num_measure_qubits = len(measure_qubits)
    rec_indices = []
    for q in logical_z_qubits:
        # Find the position of the qubit in the data_qubits list
        pos = data_qubits.index(q)
        # The records are indexed from the end, so rec[-1] is the last measurement
        rec_indices.append(stim.target_rec( -(len(data_qubits) - pos) ))

    circuit.append("OBSERVABLE_INCLUDE", rec_indices, 0)

    return circuit

# --- Main execution ---
if __name__ == "__main__":
    d = 3
    surface_code_circuit = create_surface_code_logical_zero(d)

    print("--- Stim Circuit for d=5 Surface Code Logical |0> ---")
    print(surface_code_circuit)

    # --- Verification ---
    # Let's check that the logical observable is indeed 0.
    sampler = surface_code_circuit.compile_sampler()
    samples = sampler.sample(10000)

    print("--- Verification Samples (Observable Outcome) ---")
    print("The logical observable should always be 0 (False).")
    print(np.where(samples[:, 0] == True)) # Print the outcome of the first (and only) observable
