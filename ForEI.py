import numpy as np
import matplotlib.pyplot as plt

# 参数（要和 VASP & 脚本一致）
NBAND = 16     # 换成你的 NBANDS
NPOINT = 97     # 长方形路径上总 k 点数
EF = -2.01449370  # 费米能级 eV

# 读数据
data = np.loadtxt('2.dat')    # 形状：(NPOINT, NBAND)

# 如果形状颠倒了，可以 print(data.shape) 看看，
# 但按脚本逻辑应该是 (NPOINT, NBAND)

# 减去费米面
data_shift = data - EF

# k 轴：简单用 0 到 NPOINT-1
k = np.arange(NPOINT)

plt.figure()

for ib in range(NBAND):
    plt.plot(k, data_shift[:, ib])

# 画费米能级
plt.axhline(0.0, linestyle='--')

# 限制能量范围
plt.ylim(-5, 3)
plt.ylabel('E - $E_F$ (eV)')
plt.xlabel('k-path: $\Gamma$–X–M–Y–$\Gamma$')

# 标出高对称点位置
k_GM = 0
k_X  = 20
k_M  = 48
k_Y  = 68
k_GM_end = 96

for v in [k_GM, k_X, k_M, k_Y, k_GM_end]:
    plt.axvline(v, linestyle=':', linewidth=0.8)

plt.xticks(
    [k_GM, k_X, k_M, k_Y, k_GM_end],
    [r'$\Gamma$', 'X', 'M', 'Y', r'$\Gamma$']
)

plt.tight_layout()
plt.show()
