#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
需求：
1) 在 A 中，找出“子文件夹的直系文件数 < 7”的子文件夹，打印名字并移动整个子文件夹到 B
2) 在 B 中，找出“子文件夹的直系文件数 > 7”的子文件夹，移动整个子文件夹到 A

默认非递归计数（只统计直系文件）；如需递归，可把 count_files() 中的 iterdir 改为 rglob('*') 并判断 is_file()。
"""

import argparse
import shutil
from pathlib import Path
import sys

def parse_args():
    p = argparse.ArgumentParser(description="按文件数阈值在 A 与 B 之间移动子文件夹（非递归计数）。")
    p.add_argument("--a", default=r"C:\Users\10169\Downloads\a", help="A 路径（默认：%(default)s）")
    p.add_argument("--b", default=r"C:\Users\10169\Downloads\b", help="B 路径（默认：%(default)s）")
    p.add_argument("--lt", type=int, default=7, help="A 中判定“小于”的阈值（默认：7）")
    p.add_argument("--gt", type=int, default=7, help="B 中判定“大于”的阈值（默认：7）")
    p.add_argument("--dry-run", action="store_true", help="试运行：只打印不移动")
    return p.parse_args()

def count_files_nonrecursive(dir_path: Path) -> int:
    """只统计直系文件数量（非递归）。"""
    return sum(1 for p in dir_path.iterdir() if p.is_file())

def unique_dest_dir(dest_root: Path, name: str) -> Path:
    """为文件夹生成不冲突的落地路径。"""
    cand = dest_root / name
    i = 1
    while cand.exists():
        cand = dest_root / f"{name} ({i})"
        i += 1
    return cand

def move_dir(src: Path, dest_root: Path, dry: bool, tag: str):
    target = unique_dest_dir(dest_root, src.name)
    print(f"{tag}  {src}  ->  {target}")
    if not dry:
        dest_root.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(target))

def main():
    args = parse_args()
    A = Path(args.a)
    B = Path(args.b)

    if not A.exists():
        print(f"ERROR: A 路径不存在：{A}", file=sys.stderr); sys.exit(1)
    if not B.exists():
        print(f"ERROR: B 路径不存在：{B}", file=sys.stderr); sys.exit(1)

    print(f"A = {A}")
    print(f"B = {B}")
    print(f"判定阈值：A 中 < {args.lt}；B 中 > {args.gt}")
    print(f"DRY RUN: {args.dry_run}")
    print("-" * 72)

    # 1) A 中：找出文件数 < lt 的子文件夹，打印名字并整体移到 B
    print("# 从 A -> B：子文件夹直系文件数 < 阈值 的整个文件夹")
    moved_a2b = 0
    for d in (p for p in A.iterdir() if p.is_dir()):
        fc = count_files_nonrecursive(d)
        if fc < args.lt:
            print(f"  匹配：{d.name}  (文件数={fc})")
            move_dir(d, B, args.dry_run, tag="[MOVE A->B]")
            moved_a2b += 1
    if moved_a2b == 0:
        print("  无匹配。")

    print("-" * 72)

    # 2) B 中：找出文件数 > gt 的子文件夹，整体移到 A
    print("# 从 B -> A：子文件夹直系文件数 > 阈值 的整个文件夹")
    moved_b2a = 0
    for d in (p for p in B.iterdir() if p.is_dir()):
        fc = count_files_nonrecursive(d)
        if fc > args.gt:
            print(f"  匹配：{d.name}  (文件数={fc})")
            move_dir(d, A, args.dry_run, tag="[MOVE B->A]")
            moved_b2a += 1
    if moved_b2a == 0:
        print("  无匹配。")

    print("-" * 72)
    print(f"总结：A->B 移动 {moved_a2b} 个文件夹；B->A 移动 {moved_b2a} 个文件夹。")
    print("完成。")

if __name__ == "__main__":
    main()
