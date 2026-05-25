#!/usr/bin/env python3
import os
import sys
import mmap
import multiprocessing as mp
import numpy as np

def process_chunk(chunk_data):
    """处理一个数据块（bytes对象），返回该块的sum, min, max"""
    arr = np.frombuffer(chunk_data, dtype='>u4')
    if arr.size == 0:
        return (0, None, None)
    return (int(arr.sum()), int(arr.min()), int(arr.max()))

def main():
    if len(sys.argv) != 2:
        print("Usage: python calc_data.py <input_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    if not os.path.exists(input_file):
        print(f"Error: file {input_file} not found")
        sys.exit(1)

    file_size = os.path.getsize(input_file)
    if file_size % 4 != 0:
        print("Error: file size is not a multiple of 4")
        sys.exit(1)

    num_cores = mp.cpu_count()
    print(f"Using {num_cores} processes", file=sys.stderr)

    chunk_size = (file_size // num_cores // 4) * 4
    if chunk_size == 0:
        chunk_size = 4

    chunks = []
    for i in range(num_cores):
        offset = i * chunk_size
        if i == num_cores - 1:
            size = file_size - offset
        else:
            size = chunk_size
        if size > 0:
            chunks.append((offset, size))

    def read_chunk(offset, size):
        with open(input_file, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                return mm[offset:offset+size]

    tasks = [read_chunk(off, sz) for off, sz in chunks]

    with mp.Pool(processes=num_cores) as pool:
        results = pool.map(process_chunk, tasks)

    total_sum = 0
    total_min = None
    total_max = None
    for s, mn, mx in results:
        if mn is None:
            continue
        total_sum += s
        if total_min is None or mn < total_min:
            total_min = mn
        if total_max is None or mx > total_max:
            total_max = mx

    print(f"sum={total_sum} min={total_min} max={total_max}")

if __name__ == "__main__":
    main()