import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from logcircuit import LogCircuit
# --- 1. 定义电路组件类 ---



def process_circuit(circuit):
    """
    遍历电路，计算每个探测器的最终空间坐标。

    Args:
        circuit (list): 一个包含 Detector 和 ShiftCoords 对象的列表。

    Returns:
        dict: 一个字典，键是探测器的 index，值是其最终的三维坐标 (numpy array)。
    """
    # 初始化累积位移向量
    cumulative_shift = np.array([0.0, 0.0, 0.0])
    
    # 用于存储最终结果的字典
    final_detector_positions = {}
    index = 0
    print("--- 开始处理电路 ---")
    for op in circuit:
        if op.name == "REPEAT":
            for i in range(op.repeat_count):
                for sub_op in op.body_copy():
                    if sub_op.name == "SHIFT_COORDS":
                        value = sub_op.gate_args_copy()
                        print(f"遇到位移: {value}，当前累积位移变为 {cumulative_shift + value}")
                        cumulative_shift += value
                    elif sub_op.name == "DETECTOR":
                        ini_coord = sub_op.gate_args_copy()
                        final_coord = ini_coord + cumulative_shift
                        final_detector_positions[index] = final_coord
                        print(f"探测器 {index}: 初始坐标 {ini_coord} + 累积位移 {cumulative_shift} -> 最终坐标 {final_coord}")
                        index += 1
        elif op.name == "SHIFT_COORDS":
            # 当遇到位移操作时，累加位移量
            value = op.gate_args_copy()
            print(f"遇到位移: {value}，当前累积位移变为 {cumulative_shift + value}")
            cumulative_shift += value
        elif op.name == "DETECTOR":
            # 当遇到探测器时，计算其最终坐标
            ini_coord = op.gate_args_copy()
            final_coord = ini_coord + cumulative_shift
            final_detector_positions[index] = final_coord
            print(f"探测器 {index}: 初始坐标 {ini_coord} + 累积位移 {cumulative_shift} -> 最终坐标 {final_coord}")
            index += 1
        else:
            continue
    print("--- 电路处理完成 ---")
    return final_detector_positions

def find_detector_coordinate(detector_index, final_positions):
    """
    从已处理的结果中查找给定序号的探测器的三维坐标。

    Args:
        detector_index (int): 要查询的探测器序号。
        final_positions (dict): process_circuit 函数返回的结果字典。

    Returns:
        numpy.ndarray or None: 如果找到，返回坐标；否则返回 None。
    """
    positions = {}
    for detector in detector_index:
        coords = final_positions.get(detector)
        positions[detector] = coords
    
    return positions

# --- 3. 三维可视化 ---

def plot_specific_points_with_planes_and_line(points_to_plot: dict):
    if not points_to_plot:
        print("没有可供绘制的探测器位置。")
        return
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    indices = list(points_to_plot.keys())
    coords = np.array(list(points_to_plot.values()))
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c='blue', marker='o', s=80, depthshade=True, label='探测器')
    for i, index in enumerate(indices):
        ax.text(coords[i, 0], coords[i, 1], coords[i, 2], f' {index}', color='red', fontsize=12, zorder=10)
    if len(coords) > 1:
        ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], c='green', linestyle='-', marker='', label='路径')
    if len(coords) > 0:
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        x_range = np.linspace(x_min - 5, x_max + 5, 2)
        y_range = np.linspace(y_min - 5, y_max + 5, 2)
        X_plane, Y_plane = np.meshgrid(x_range, y_range)
        z_min = np.floor(coords[:, 2].min())
        z_max = np.ceil(coords[:, 2].max())
        
        for z_level in range(int(z_min), int(z_max) + 1):
            Z_plane = np.full_like(X_plane, z_level)
            ax.plot_surface(X_plane, Y_plane, Z_plane, alpha=0.15, color='green', rstride=1, cstride=1, edgecolor='k', linewidth=0.2)
    ax.set_xlabel('X 坐标', fontsize=12)
    ax.set_ylabel('Y 坐标', fontsize=12)
    ax.set_zlabel('Z 坐标', fontsize=12)
    ax.set_title('特定探测器位置、路径及Z平面', fontsize=16)
    ax.legend()
    ax.grid(True)
    # plt.show()
    plt.savefig('detector_positions_with_planes_and_line.png', dpi=300)


# --- 主程序：定义电路并执行 ---

if __name__ == "__main__":
    # 1. 定义一个示例电路
    # 电路是按顺序执行的操作列表
    p = (0.0001, 0.001) ## Pre-set parameters for error rate
    distance = 5
    rounds = 2
    LogCir = LogCircuit(distance, rounds, p, 2)
    LogCir.err_free_prep()
    LogCir.log_circ_prep("CX")
    LogCir.qec_block(guide = "CX")
    LogCir.log_circ_prep("CX")
    LogCir.qec_block()
    LogCir.virtual_obs_log()
    print(LogCir.circuit)
    # 2. 处理电路，得到所有探测器的最终位置
    final_positions = process_circuit(LogCir.circuit)
    # exit(0)
    # 3. 查询指定序号的探测器坐标
    index_to_find = [128, 150, 162, 168, 247, 249]
    
    found_coord = find_detector_coordinate(index_to_find, final_positions)
    print(found_coord)
    


    # 4. 绘制三维图像
    plot_specific_points_with_planes_and_line(found_coord)

