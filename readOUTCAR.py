import numpy as np
import matplotlib.pyplot as plt
import re
import os

# ===== 可调参数 =====
E_MIN = -5.0   # y 轴下限 (eV)
E_MAX =  3.0   # y 轴上限 (eV)
EIG_FILE = "EIGENVAL"
OUTCAR_FILE = "OUTCAR"  # 如果没有，就用 EF=0 画

def read_eigenval(filename=EIG_FILE):
    """
    直接解析 VASP 的 EIGENVAL 文件
    返回：
      kpts: (NKPTS, 3) 的 k 点坐标数组
      energies: (NKPTS, NBANDS) 的能量数组 (eV)
    """
    with open(filename, "r") as f:
        # 前 5 行是头信息，不用管
        for _ in range(5):
            next(f)
        # 第 6 行:   NIONS  NKPTS  NBANDS
        line = next(f)
        parts = line.split()
        n_ions, n_kpts, n_bands = map(int, parts)
        # 第 7 行一般是空行，读掉
        _ = next(f)

        kpts = np.zeros((n_kpts, 3), dtype=float)
        energies = np.zeros((n_kpts, n_bands), dtype=float)

        for ik in range(n_kpts):
            # 读 k 点行（可能前面有空行，所以跳过空行）
            line = next(f)
            while not line.strip():
                line = next(f)
            kvec = list(map(float, line.split()[:3]))
            kpts[ik, :] = kvec

            # 下面紧跟 NBANDS 行，每行是: band_index  energy  occupation
            for ib in range(n_bands):
                line = next(f)
                while not line.strip():
                    line = next(f)
                sp = line.split()
                energies[ik, ib] = float(sp[1])

        return kpts, energies

def read_fermi(filename=OUTCAR_FILE, default_ef=-2.01449370):
    """
    从 OUTCAR 里读费米能级 E-fermi。
    找不到就返回 default_ef。
    """
    if not os.path.exists(filename):
        print(f"[警告] {filename} 不存在，将使用 EF = {default_ef:.3f} eV")
        return default_ef

    ef = None
    pattern = re.compile(r"E-fermi\s*:\s*([-\d\.Ee+]+)")
    with open(filename, "r") as f:
        for line in f:
            if "E-fermi" in line:
                m = pattern.search(line)
                if m:
                    ef = float(m.group(1))
    if ef is None:
        print(f"[警告] 在 {filename} 中没有找到 E-fermi，将使用 EF = {default_ef:.3f} eV")
        ef = default_ef
    else:
        print(f"[信息] 从 {filename} 读取到 E-fermi = {ef:.6f} eV")
    return ef

def build_kpath(kpts):
    """
    根据 k 点坐标计算累计距离，用于 x 轴:
      s_0 = 0
      s_i = sum |k_i - k_{i-1}|
    """
    n_kpts = kpts.shape[0]
    kdist = np.zeros(n_kpts, dtype=float)
    for i in range(1, n_kpts):
        dk = kpts[i] - kpts[i-1]
        kdist[i] = kdist[i-1] + np.linalg.norm(dk)
    return kdist

def main():
    # 1. 读 EIGENVAL
    print("[信息] 正在从 EIGENVAL 读取能带数据...")
    kpts, energies = read_eigenval(EIG_FILE)
    n_kpts, n_bands = energies.shape
    print(f"[信息] 读取到 {n_kpts} 个 k 点, {n_bands} 条能带")

    # 2. 读费米能级
    ef = read_fermi(OUTCAR_FILE, default_ef=0.0)

    # 3. 构造 k-path 横坐标
    kdist = build_kpath(kpts)

    # 4. 所有能量减去费米面
    energies_shift = energies - ef

    # 5. 画图
    plt.figure(figsize=(5, 6))
    for ib in range(n_bands):
        plt.plot(kdist, energies_shift[:, ib])

    # 画费米面 (E - Ef = 0)
    plt.axhline(0.0, linestyle="--", linewidth=0.8)

    # y 轴范围
    plt.ylim(E_MIN, E_MAX)

    plt.xlabel("k-path")
    plt.ylabel("E - $E_F$ (eV)")

    # 如果你知道这是 Γ->X，可以手动在中间加一个竖线
    # 例如：X 在最后一个点 ->:
    # plt.axvline(kdist[-1], linestyle=":", linewidth=0.8)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
