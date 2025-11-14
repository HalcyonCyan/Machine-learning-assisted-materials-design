import os
import re
import matplotlib.pyplot as plt
import numpy as np


def parse_folder_name(folder_name):
    """从文件夹名中提取a和b的值"""
    match = re.search(r'a1_([\d.]+)_a2_([\d.]+)', folder_name)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None


def check_log_file(log_path):
    """检查log文件的最后一行是否符合条件"""
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            if lines:
                last_line = lines[-1].strip()
                return "reached required accuracy - stopping structural energy minimisation" in last_line
    except:
        pass
    return False


def extract_energy_from_outcar(outcar_path):
    """从OUTCAR文件的末尾50行提取能量值"""
    try:
        with open(outcar_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            # 读取最后50行
            last_lines = lines[-150:] if len(lines) >= 150 else lines

            for line in reversed(last_lines):
                if "free  energy   TOTEN" in line:
                    # 提取能量值
                    match = re.search(r'free  energy   TOTEN  =\s+([-\d.]+)', line)
                    if match:
                        return float(match.group(1))
    except:
        pass
    return None


def main():
    base_path = r'C:\Users\10169\Desktop\22'

    # 存储数据
    a_values = []
    b_values = []
    energy_values = []
    status_colors = []  # 存储点的颜色
    valid_energies = []  # 仅存储有效的能量值
    folder_names = []  # 存储文件夹名称
    valid_folders = []  # 存储有效数据对应的文件夹

    # 遍历所有文件夹
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)

        if os.path.isdir(folder_path):
            a_val, b_val = parse_folder_name(folder_name)

            if a_val is not None and b_val is not None:
                log_path = os.path.join(folder_path, 'log')
                outcar_path = os.path.join(folder_path, 'OUTCAR')

                # 检查log文件
                log_valid = check_log_file(log_path)

                if log_valid and os.path.exists(outcar_path):
                    # 提取能量值
                    energy = extract_energy_from_outcar(outcar_path)

                    if energy is not None:
                        a_values.append(a_val)
                        b_values.append(b_val)
                        energy_values.append(energy)
                        valid_energies.append(energy)
                        folder_names.append(folder_name)
                        valid_folders.append(folder_name)
                        status_colors.append('blue')  # 有效数据用蓝色
                    else:
                        a_values.append(a_val)
                        b_values.append(b_val)
                        energy_values.append(None)
                        folder_names.append(folder_name)
                        status_colors.append('gray')  # log有效但能量提取失败
                else:
                    a_values.append(a_val)
                    b_values.append(b_val)
                    energy_values.append(None)
                    folder_names.append(folder_name)
                    status_colors.append('gray')  # log无效或OUTCAR不存在

    # 创建绘图
    plt.figure(figsize=(12, 8))

    # 绘制散点图
    for i in range(len(a_values)):
        if status_colors[i] == 'blue':
            plt.scatter(a_values[i], b_values[i], c=[energy_values[i]],
                        cmap='viridis', s=50, alpha=0.7)
        else:
            plt.scatter(a_values[i], b_values[i], color='gray', s=30, alpha=0.5)

    # 添加颜色条
    if valid_energies:
        plt.colorbar(label='Free Energy (eV)')

    plt.xlabel('a value')
    plt.ylabel('b value')
    plt.title('2D Energy Distribution (Gray points: did not reach required accuracy)')
    plt.grid(True, alpha=0.3)

    # 输出最大最小值及其位置
    if valid_energies:
        min_energy = min(valid_energies)
        max_energy = max(valid_energies)

        # 找到最小值和最大值对应的索引
        min_indices = [i for i, energy in enumerate(energy_values) if energy == min_energy]
        max_indices = [i for i, energy in enumerate(energy_values) if energy == max_energy]

        print(f"所有OUTCAR文件中free energy的最小值: {min_energy:.8f} eV")
        for idx in min_indices:
            print(f"  出现在文件夹: {folder_names[idx]}, 坐标: a={a_values[idx]:.4f}, b={b_values[idx]:.4f}")

        # 在图中标注最大最小值
        # 只标注第一个最小值点和第一个最大值点，避免图中过于拥挤
        if min_indices:
            min_idx = min_indices[0]
            plt.annotate(f'Min: {min_energy:.4f}eV',
                         xy=(a_values[min_idx], b_values[min_idx]),
                         xytext=(10, 10), textcoords='offset points',
                         bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7),
                         arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))



        # 如果有多个相同的最小值或最大值，在控制台额外提示
        if len(min_indices) > 1:
            print(f"注意: 有 {len(min_indices)} 个文件夹具有相同的最小能量值")

        if len(max_indices) > 1:
            print(f"注意: 有 {len(max_indices)} 个文件夹具有相同的最大能量值")
    else:
        print("未找到有效的能量数据")

    plt.tight_layout()
    plt.show()

    # 打印统计信息
    total_folders = len(a_values)
    valid_count = sum(1 for color in status_colors if color == 'blue')
    gray_count = sum(1 for color in status_colors if color == 'gray')

    print(f"\n统计信息:")
    print(f"总文件夹数: {total_folders}")
    print(f"有效数据点: {valid_count}")
    print(f"无效/灰色点: {gray_count}")


if __name__ == "__main__":
    main()