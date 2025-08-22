import stim
import numpy as np
import pymatching
import matplotlib.pyplot as plt
def memory_experiment(circuit, N):
    sampler = circuit.compile_detector_sampler()

    shots = N

    detection_events, actual_observable_flips = sampler.sample(shots, separate_observables=True)
    error_model = circuit.detector_error_model()

    matching = pymatching.Matching.from_detector_error_model(error_model)

    predicted_observable_flips = matching.decode_batch(detection_events)

    num_log_errors = np.sum(actual_observable_flips != predicted_observable_flips)

    logical_error_rate = num_log_errors / shots


    return logical_error_rate
# def surface_code_modified(distance, rounds, depo, before_round, flip):
#     """Create a stim Circuit that measures X and Z stabilizers at the same time"""
#     circuit_x = stim.Circuit.generated("surface_code:rotated_memory_x", distance=distance,
#                                        rounds = rounds, after_clifford_depolarization=depo, 
#                                        before_round_data_depolarization=before_round,
#                                        before_measure_flip_probability=flip)

#     circuit_z = stim.Circuit.generated("surface_code:rotated_memory_z", distance=distance,
#                                        rounds = rounds, after_clifford_depolarization=depo, 
#                                        before_round_data_depolarization=before_round,
#                                        before_measure_flip_probability=flip)
#     virtual_x, virtual_z = circuit_x[-D**2//2 - 2:], circuit_z[-D**2//2 - 2:]
#     repeatblock = circuit_x[-D**2//2 - 3]
#     assert isinstance(repeatblock, stim.CircuitRepeatBlock)
#     init = circuit_x[:-2*(D**2//2) - 4]
#     detect_x = circuit_x[-2*(D**2//2) - 4:-D**2//2 - 3]
#     detect_z = circuit_z[-2*(D**2//2) - 4:-D**2//2 - 3]
#     # print(virtual_x, virtual_z)
    
#     real_circuit_x = init + detect_x + detect_z  
#     real_circuit_z = init + detect_x + detect_z
#     real_circuit_x.append(repeatblock)
#     real_circuit_z.append(repeatblock)
#     real_circuit_x += virtual_x
#     real_circuit_z += virtual_z

#     return real_circuit_x, real_circuit_z

if __name__ == "__main__":

    D = 3
    depo = 0.006
    before_round = 0.001
    flip = 0.001
    
   
    
    Xp = [0.003 * (0.01/0.003)**(i/7) for i in range(8) ]
    Ler = np.zeros((4, 8))
    N = 1000000
    
    ## Experiment I. LER vs PER for every Xp and distance
    for D in range(3, 10, 2):
        for i in range(8):
            circuit = stim.Circuit.generated("surface_code:rotated_memory_z", 
                                             distance = D, after_clifford_depolarization=Xp[i],
                                              before_measure_flip_probability=0.001,rounds = D)
            Ler[(D-3) // 2, i] = round(memory_experiment(circuit, N), 5)
    print("LER vs PER at different distances{3,5,7,9}")
    print(Ler)
    print("---------------")
    ## Experiment II. LER vs PER for every Xp and different rounds (distance = 5)
    Ler_rounds = np.zeros((4,5))
    Ler_rounds2 = np.zeros((4, 5))
    for i in range(4):
        for j in range(5):
            circuit = stim.Circuit.generated("surface_code:rotated_memory_z", 
                                             distance = 5, after_clifford_depolarization=Xp[i],
                                              before_measure_flip_probability=0.001,rounds = j+1)
            Ler_rounds2[i, j] = round(memory_experiment(circuit, N), 5)
    # for i in range(4):
    #     for j in range(5):
    #         circuit = stim.Circuit.generated("surface_code:rotated_memory_z", 
    #                                          distance = 5, before_round_data_depolarization = Xp[i],
    #                                           before_measure_flip_probability=0.001,rounds = j+1)
            Ler_rounds[i, j] = memory_experiment(circuit, N)
    print("LER vs measurement rounds at PER {0.003-0.042}")
    print(Ler_rounds2)
    
   
    # plt.figure(figsize=(10, 6))
    # plt.title("Logical Error Rate vs Distance (Free)")
    # plt.xlabel("Distance")
    # plt.ylabel("Logical Error Rate")
    # plt.grid(True)
    # plt.yscale('log')
    # plt.xscale('linear')
    # for i in range(4):
    #     plt.plot(Xp, Ler[i], label=f"d = {2 * i + 3}")
    # plt.legend()
    # plt.savefig('Output/Figures/LER_vs_Xp.png')
    # fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # axs[0].set_title("Logical Error Rate vs Measure Rounds(clifford)")
    # axs[0].set_xlabel("Distance")
    # axs[0].set_ylabel("Logical Error Rate")
    # axs[0].grid(True)
    # axs[0].set_yscale('log')
    # axs[0].set_xscale('linear')
    # for i in range(4):
    #     axs[0].plot(np.arange(1, 6), Ler_rounds2[i], label=f"p = {Xp[i]:.4f}")
    # axs[0].legend()

    # axs[1].set_title("Logical Error Rate vs Measure Rounds(before round)")
    # axs[1].set_xlabel("Distance")
    # axs[1].set_ylabel("Logical Error Rate")
    # axs[1].grid(True)
    # axs[1].set_yscale('log')
    # axs[1].set_xscale('linear')
    # for i in range(4):
    #     axs[1].plot(np.arange(1, 6), Ler_rounds[i], label=f"p = {Xp[i]:.4f}")
    # axs[1].legend()

    # plt.figure(figsize=(10, 6))
    # plt.title("Logical Error Rate vs Measure Rounds")
    # plt.xlabel("Distance")
    # plt.ylabel("Logical Error Rate")
    # plt.grid(True)
    # plt.yscale('log')
    # plt.xscale('linear')
    # for i in range(4):
    #     plt.plot(np.arange(1, 6), Ler_rounds[i], label=f"p = {Xp[i]:.4f}")
    # plt.legend()
    # plt.savefig('Output/Figures/LER_vs_rounds.png')
                    
