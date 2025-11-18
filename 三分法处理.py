#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
二维 (a,b) 三分搜索 + VASP 计算能量

功能概述：
- 给定一个“基准目录 base_dir”，里面有可运行 VASP 的 INCAR / POSCAR / POTCAR / KPOINTS 等；
- 给定 a 区间 [a_left, a_right]，b 区间 [b_left, b_right]；
- 对 a 做外层三分，对每个候选 a，内部对 b 再做一次三分；
- 每次需要计算 f(a,b) 时：
    * 在 work_root 下创建子目录 a{a值}_b{b值}
    * 从 base_dir 拷贝所有文件到该子目录
    * 修改 POSCAR 第 3 行第 1 个数为 a，第 4 行第 2 个数为 b
    * 在该目录下运行 VASP（同步等待结束）
    * 从 OUTCAR 中提取 "free  energy   TOTEN" 作为能量
- 最终输出近似最优 (a*, b*, E_min)，并把所有计算过的点写入 CSV 文件。

使用示例：
    python scan_ab_ternary.py \\
        --base-dir /home/student/workspace/MLBP/base_vasp \\
        --work-root /home/student/workspace/MLBP/ternary_runs \\
        --a-left 4.3 --a-right 4.9 \\
        --b-left 3.1 --b-right 3.5

可以根据需要修改脚本顶部的 VASP_CMD / TOL_A / TOL_B 等参数。
"""

import os
import re
import shutil
import time
import math
import argparse
import subprocess
from typing import Tuple, Dict

# ========= 可调参数（你可以在这里根据实际环境修改） =========

# VASP 运行命令（同步执行）
# 如果想完全照你平时命令，可以改成： "mpirun -np 4 vasp.std.6.5.1"
# 不建议带 nohup 和 &，会变成后台异步，三分无法等待结果。
VASP_CMD = "mpirun -np 4 vasp.std.6.5.1"

# 三分精度 & 最大迭代次数
TOL_A = 0.005     # a 方向精度
TOL_B = 0.005     # b 方向精度
MAX_ITER_A = 8    # a 方向最多三分迭代次数
MAX_ITER_B = 8    # b 方向最多三分迭代次数

# 是否重用已有 OUTCAR（如果该点已经算过就不再重新跑 VASP）
REUSE_EXISTING = True

# ====================================================


def extract_energy_from_outcar(outcar_path: str):
    """从 OUTCAR 文件的末尾若干行提取 TOTEN 能量值。"""
    try:
        with open(outcar_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            last_lines = lines[-150:] if len(lines) >= 150 else lines

            for line in reversed(last_lines):
                if "free  energy   TOTEN" in line:
                    match = re.search(r'free  energy   TOTEN  =\s+([-\d.]+)', line)
                    if match:
                        return float(match.group(1))
    except Exception as e:
        print(f"[WARN] 读取 OUTCAR 失败: {outcar_path} ({e})")
    return None


def modify_poscar_for_ab(poscar_path: str, a_val: float, b_val: float):
    """
    修改 POSCAR：
    - 第 3 行第 1 个数 = a_val
    - 第 4 行第 2 个数 = b_val
    其他保持不变。
    """
    with open(poscar_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    if len(lines) < 4:
        raise RuntimeError(f"POSCAR 行数不足：{poscar_path}")

    # 行号：0 标题；1 缩放；2 为第三行；3 为第四行
    line3 = lines[2].split()
    line4 = lines[3].split()

    if len(line3) < 1 or len(line4) < 2:
        raise RuntimeError(f"POSCAR 第三/四行格式不正确：{poscar_path}")

    line3[0] = f"{a_val:.16f}"
    line4[1] = f"{b_val:.16f}"

    lines[2] = "  ".join(line3) + "\n"
    lines[3] = "  ".join(line4) + "\n"

    with open(poscar_path, 'w', encoding='utf-8', errors='ignore') as f:
        f.writelines(lines)


def ensure_run_dir(base_dir: str, work_root: str, a_val: float, b_val: float) -> str:
    """
    创建并返回当前 (a,b) 对应的计算目录：
    - 名字格式：a{a_val}_b{b_val}，例如：a4.5800_b3.3200
    - 如果目录不存在：从 base_dir 拷贝所有文件。
    """
    dir_name = f"a{a_val:.4f}_b{b_val:.4f}"
    run_dir = os.path.join(work_root, dir_name)
    os.makedirs(work_root, exist_ok=True)

    if not os.path.exists(run_dir):
        print(f"[INFO] 创建新计算目录: {run_dir}")
        shutil.copytree(base_dir, run_dir)
    else:
        # 目录已经存在，不再整体复制
        print(f"[INFO] 使用已有目录: {run_dir}")

    # 修改 POSCAR
    poscar_path = os.path.join(run_dir, "POSCAR")
    if not os.path.exists(poscar_path):
        raise FileNotFoundError(f"POSCAR 不存在: {poscar_path}")

    modify_poscar_for_ab(poscar_path, a_val, b_val)
    return run_dir


def run_vasp_and_get_energy(base_dir: str,
                            work_root: str,
                            a_val: float,
                            b_val: float) -> float:
    """
    对给定 (a,b) 调用 VASP 计算，并返回 TOTEN。
    - 会在 work_root/a{a}_b{b} 目录下运行；
    - 重用已有 OUTCAR（如果 REUSE_EXISTING = True 且能提取到能量）。
    """
    run_dir = ensure_run_dir(base_dir, work_root, a_val, b_val)
    outcar_path = os.path.join(run_dir, "OUTCAR")

    if REUSE_EXISTING and os.path.exists(outcar_path):
        E = extract_energy_from_outcar(outcar_path)
        if E is not None:
            print(f"[INFO] 重用已有能量 E = {E:.8f} (a={a_val:.4f}, b={b_val:.4f})")
            return E
        else:
            print(f"[INFO] 已有 OUTCAR 无法提取能量，将重新计算：{outcar_path}")

    # 如果存在旧的 OUTCAR，可以删掉，避免混淆
    if os.path.exists(outcar_path):
        os.remove(outcar_path)

    print(f"[RUN] 运行 VASP at a={a_val:.6f}, b={b_val:.6f} in {run_dir}")
    # 同步运行 VASP，输出重定向到 log
    with open(os.path.join(run_dir, "log"), "w") as logf:
        # 这里不使用 nohup 和 &，保证脚本在 VASP 结束前不会继续往下走
        proc = subprocess.run(
            VASP_CMD.split(),
            cwd=run_dir,
            stdout=logf,
            stderr=subprocess.STDOUT
        )

    if proc.returncode != 0:
        raise RuntimeError(f"VASP 运行失败 (returncode={proc.returncode}) in {run_dir}")

    # VASP 正常结束后，读取 OUTCAR
    E = extract_energy_from_outcar(outcar_path)
    if E is None:
        raise RuntimeError(f"无法从 OUTCAR 提取能量: {outcar_path}")

    print(f"[OK ] E = {E:.8f} (a={a_val:.6f}, b={b_val:.6f})")
    return E


class EnergyCache:
    """
    计算 f(a,b) 的缓存：避免同一个点重复跑 VASP。
    """

    def __init__(self, base_dir: str, work_root: str):
        self.base_dir = base_dir
        self.work_root = work_root
        self.cache: Dict[Tuple[float, float], float] = {}

    def get_energy(self, a: float, b: float) -> float:
        # 可以把 key 做一点 round，避免浮点小数微小差异导致重复
        key = (round(a, 6), round(b, 6))
        if key in self.cache:
            return self.cache[key]
        E = run_vasp_and_get_energy(self.base_dir, self.work_root, a, b)
        self.cache[key] = E
        return E


def ternary_search_b_for_fixed_a(a: float,
                                 b_left: float,
                                 b_right: float,
                                 cache: EnergyCache,
                                 tol_b: float,
                                 max_iter_b: int):
    """
    在固定 a 的情况下，对 b 做一维三分，找到近似最优 b 和对应能量。
    """
    L = b_left
    R = b_right
    for it in range(max_iter_b):
        if abs(R - L) < tol_b:
            break
        b1 = (2.0 * L + R) / 3.0
        b2 = (L + 2.0 * R) / 3.0
        E1 = cache.get_energy(a, b1)
        E2 = cache.get_energy(a, b2)
        print(f"[inner {it}] a={a:.6f}, b1={b1:.6f}, E1={E1:.8f}, b2={b2:.6f}, E2={E2:.8f}")
        if E1 < E2:
            R = b2
        else:
            L = b1

    b_opt = (L + R) / 2.0
    E_opt = cache.get_energy(a, b_opt)
    return b_opt, E_opt


def ternary_search_2d(a_left: float,
                      a_right: float,
                      b_left: float,
                      b_right: float,
                      cache: EnergyCache,
                      tol_a: float,
                      tol_b: float,
                      max_iter_a: int,
                      max_iter_b: int):
    """
    外层对 a 三分，内层对 b 三分，实现二维 (a,b) 搜索。
    返回：best_a, best_b, best_E, history_list
    其中 history_list 中记录每次外层迭代的若干信息，便于后处理/作图。
    """
    L = a_left
    R = a_right
    history = []

    for it in range(max_iter_a):
        if abs(R - L) < tol_a:
            break

        a1 = (2.0 * L + R) / 3.0
        a2 = (L + 2.0 * R) / 3.0

        print(f"\n[outer {it}] a1={a1:.6f}, a2={a2:.6f}")

        b1_opt, E1 = ternary_search_b_for_fixed_a(
            a1, b_left, b_right, cache, tol_b, max_iter_b
        )
        b2_opt, E2 = ternary_search_b_for_fixed_a(
            a2, b_left, b_right, cache, tol_b, max_iter_b
        )

        history.append({
            "iter": it,
            "a1": a1, "b1": b1_opt, "E1": E1,
            "a2": a2, "b2": b2_opt, "E2": E2,
            "L": L, "R": R
        })

        print(f"[outer {it}] result: a1={a1:.6f}, b1={b1_opt:.6f}, E1={E1:.8f}; "
              f"a2={a2:.6f}, b2={b2_opt:.6f}, E2={E2:.8f}")

        if E1 < E2:
            R = a2
        else:
            L = a1

    # 最后在 [L,R] 中间取一点，再做一次 b 三分，作为最终结果
    a_best = (L + R) / 2.0
    b_best, E_best = ternary_search_b_for_fixed_a(
        a_best, b_left, b_right, cache, tol_b, max_iter_b
    )

    return a_best, b_best, E_best, history


def save_results_csv(cache: EnergyCache, csv_path: str):
    """
    把所有已经计算过的 (a,b,E) 存到 CSV 文件，便于画图/分析。
    """
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("a,b,E\n")
        for (a, b), E in sorted(cache.cache.items()):
            f.write(f"{a:.6f},{b:.6f},{E:.12f}\n")
    print(f"[INFO] 所有计算点已写入: {csv_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="二维 (a,b) 三分搜索 + VASP 计算脚本")

    parser.add_argument("--base-dir", required=True,
                        help="包含 INCAR/POSCAR/POTCAR/KPOINTS 的基准目录")
    parser.add_argument("--work-root", required=True,
                        help="实际运行计算的工作根目录，子目录会在这里创建")

    parser.add_argument("--a-left", type=float, required=True,
                        help="a 的左端点")
    parser.add_argument("--a-right", type=float, required=True,
                        help="a 的右端点")
    parser.add_argument("--b-left", type=float, required=True,
                        help="b 的左端点")
    parser.add_argument("--b-right", type=float, required=True,
                        help="b 的右端点")

    parser.add_argument("--tol-a", type=float, default=TOL_A,
                        help="a 方向三分精度")
    parser.add_argument("--tol-b", type=float, default=TOL_B,
                        help="b 方向三分精度")
    parser.add_argument("--max-iter-a", type=int, default=MAX_ITER_A,
                        help="a 方向三分最大迭代次数")
    parser.add_argument("--max-iter-b", type=int, default=MAX_ITER_B,
                        help="b 方向三分最大迭代次数")

    parser.add_argument("--csv", type=str, default="ab_energy_points.csv",
                        help="输出 (a,b,E) CSV 文件名（保存在 work_root 下）")

    return parser.parse_args()


def main():
    args = parse_args()

    base_dir = os.path.abspath(args.base_dir)
    work_root = os.path.abspath(args.work_root)

    print(f"[INFO] base_dir  = {base_dir}")
    print(f"[INFO] work_root = {work_root}")
    print(f"[INFO] a in [{args.a_left}, {args.a_right}], "
          f"b in [{args.b_left}, {args.b_right}]")

    if not os.path.isdir(base_dir):
        raise NotADirectoryError(f"基准目录不存在: {base_dir}")

    cache = EnergyCache(base_dir, work_root)

    a_best, b_best, E_best, history = ternary_search_2d(
        args.a_left, args.a_right,
        args.b_left, args.b_right,
        cache,
        tol_a=args.tol_a,
        tol_b=args.tol_b,
        max_iter_a=args.max_iter_a,
        max_iter_b=args.max_iter_b
    )

    print("\n================ 最终结果 ================")
    print(f"a_best = {a_best:.6f}")
    print(f"b_best = {b_best:.6f}")
    print(f"E_min  = {E_best:.12f}")
    print("========================================\n")

    # 保存所有计算过的点
    csv_path = os.path.join(work_root, args.csv)
    save_results_csv(cache, csv_path)


if __name__ == "__main__":
    main()

