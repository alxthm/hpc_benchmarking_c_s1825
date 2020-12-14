import argparse
import time
from typing import Sequence

import numpy as np
import torch


def norm_python(U: Sequence[float]):
    s = 0
    for i in range(len(U)):
        s += np.sqrt(np.abs(U[i]))
    return s


def norm_numpy(U: np.array):
    return np.sum(np.sqrt(np.abs(U)))


def norm_torch(U: torch.Tensor):
    return U.abs().sqrt().sum()


def quick_run(U, U_torch, n):
    t0 = time.time()
    s_python = norm_python(U)
    t1 = time.time()
    dt_python = (t1 - t0) * 1000  # ms

    t0 = time.time()
    s_numpy = norm_numpy(U)
    t1 = time.time()
    dt_numpy = (t1 - t0) * 1000  # ms

    t0 = time.time()
    s_torch = norm_torch(U_torch)
    t1 = time.time()
    dt_torch = (t1 - t0) * 1000  # ms

    print(f'python: {s_python}, {dt_python:.5f}')
    print(f'numpy: {s_numpy}, {dt_numpy:.5f}')
    print(f'torch: {s_torch}, {dt_torch:.5f}')


def run_trials(U, U_torch, n):
    print(f'n,nb_threads,mode,trial,sum,time_ms')
    modes = ['numpy', 'pytorch']
    if n < 100000:
        modes += ['python']
    for m in modes:
        for i in range(5):
            if m == 'python':
                t0 = time.time()
                s = norm_python(U)
                t1 = time.time()
            elif m == 'numpy':
                t0 = time.time()
                s = norm_numpy(U)
                t1 = time.time()
            elif m == 'pytorch':
                t0 = time.time()
                s = norm_torch(U_torch)
                t1 = time.time()
            dt = (t1 - t0) * 1000  # ms
            print(f'{n},-1,{m},{i},{s},{dt}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int)
    parser.add_argument('--run_trials', action='store_true')
    args = parser.parse_args()

    U = np.loadtxt(f'array_{args.n}.csv', dtype=float)
    U_torch = torch.from_numpy(U)

    if args.run_trials:
        run_trials(U, U_torch, args.n)
    else:
        quick_run(U, U_torch, args.n)


if __name__ == '__main__':
    main()
