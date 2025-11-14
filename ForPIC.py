import numpy as np
import matplotlib.pyplot as plt

# ===== 可调参数 =====
# 四段能带各自的费米能级 (eV)，按 [GMX, XM, MY, YGM] 的顺序写
EF_LIST = [
    -2.05449370,  # 段1: GMX.dat 的 EF
    -2.00000000,  # 段2: XM.dat  的 EF
    -2.04000000,  # 段3: MY.dat  的 EF
    -2.01881553 ,  # 段4: YGM.dat 的 EF
]

E_MIN = -5.0       # y 轴下限 (eV)
E_MAX =  3.0       # y 轴上限 (eV)

# 四段能带数据文件名（按 Γ→X→M→Y→Γ 顺序）
files = ["GMX.dat", "XM.dat", "MY.dat", "YGM.dat"]


def load_segment(fname):
    """读取一个段的能带数据：shape = (N_kpts, N_bands)"""
    data = np.loadtxt(fname)
    # 防止只有一行时变成一维
    if data.ndim == 1:
        data = data[:, None]
    return data


def main():
    # 0. 简单检查：EF_LIST 长度要和段数一致
    if len(EF_LIST) != len(files):
        raise ValueError("EF_LIST 长度必须等于 files 段数！")

    # 1. 读四段
    segs = [load_segment(f) for f in files]
    lens = [s.shape[0] for s in segs]       # 每一段的 k 点数
    nbands = segs[0].shape[1]

    # 检查每段带数是否一样
    for s in segs[1:]:
        if s.shape[1] != nbands:
            raise ValueError("四个 .dat 的能带数不一致，请检查 grep_band.sh 的 NBAND 设置")

    # 2. 拼接 k 轴（简单用等间距 index）
    total_kpts = sum(lens)
    k = np.arange(total_kpts)

    # 高对称点在整条路径上的 index
    # Γ: 起点 0
    # X: Γ→X 段最后一个点
    # M: 再加上一段 XM
    # Y: 再加上一段 MY
    # Γ(终点): 总长度-1
    idx_G = 0
    idx_X = lens[0] - 1
    idx_M = lens[0] + lens[1] - 1
    idx_Y = lens[0] + lens[1] + lens[2] - 1
    idx_G2 = total_kpts - 1

    high_sym_positions = [idx_G, idx_X, idx_M, idx_Y, idx_G2]
    high_sym_labels = [r"$\Gamma$", "X", "M", "Y", r"$\Gamma$"]

    # 3. 拼接能带并减去各自段的费米面
    plt.figure(figsize=(9, 5))  # 横向长一点

    for ib in range(nbands):
        # 对于这一条带，四段分别减去各自的 EF，再拼起来
        y_pieces = [
            segs[i][:, ib] - EF_LIST[i]
            for i in range(len(segs))
        ]
        y_concat = np.concatenate(y_pieces)
        plt.plot(k, y_concat)

    # 4. 画“整体参考费米面” y=0（这时 0 只是你平移后的参考线）
    plt.axhline(0.0, linestyle="--", linewidth=0.8)

    # 5. y 轴范围
    plt.ylim(E_MIN, E_MAX)
    plt.ylabel("E - $E_F$ (eV)")

    # 6. 在 Γ, X, M, Y, Γ 处画竖直虚线 + 只显示这些刻度
    for x in high_sym_positions:
        plt.axvline(x, linestyle=":", linewidth=0.8)

    plt.xticks(high_sym_positions, high_sym_labels)  # x轴只有符号，没有数字
    plt.xlabel("")

    # x 轴稍微留一点边
    margin = 0.02 * (k[-1] - k[0])
    plt.xlim(k[0] - margin, k[-1] + margin)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
