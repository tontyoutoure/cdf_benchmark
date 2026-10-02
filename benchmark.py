#!/usr/bin/env python3
import argparse
import gc
import statistics
import time

import numpy as np

from cdfbench import NestedCDFHolder, FlatSTLHolder, FlatNumpyHolder


def compute_image_3D_CDF(array):
    """Mirror the original Python implementation, input order Z,Y,X."""
    array = np.asarray(array)
    array = array / np.sum(array)

    sumx = np.sum(array, axis=2)
    sumxy = np.sum(sumx, axis=1)

    cdf_x = []
    for i in range(array.shape[0]):
        cdf_x.append([])
        for j in range(array.shape[1]):
            t = np.cumsum(array[i][j])
            if t[-1] != 0:
                t = t / t[-1]
            cdf_x[i].append(t)

    cdf_y = []
    for i in range(len(sumx)):
        t = np.cumsum(sumx[i])
        if t[-1] != 0:
            t = t / t[-1]
        cdf_y.append(t)

    cdf_z = np.cumsum(sumxy) / np.sum(sumxy)
    return cdf_x, cdf_y, cdf_z


def to_flat_arrays(cdf_x, cdf_y, cdf_z):
    # These conversions are intentionally timed separately.  In a redesigned
    # implementation you would ideally generate these arrays flat from the start.
    t0 = time.perf_counter_ns()
    x = np.ascontiguousarray(np.asarray(cdf_x, dtype=np.float64)).reshape(-1)
    y = np.ascontiguousarray(np.asarray(cdf_y, dtype=np.float64)).reshape(-1)
    z = np.ascontiguousarray(np.asarray(cdf_z, dtype=np.float64)).reshape(-1)
    t1 = time.perf_counter_ns()
    return (x, y, z), (t1 - t0)


def run_once(holder_cls, args_xyz, n, mode):
    holders = [holder_cls() for _ in range(n)]
    copy_ns = 0

    gc.collect()
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        t0 = time.perf_counter_ns()
        z, y, x = args_xyz
        for h in holders:
            h.set_cdf(z, y, x)
        t1 = time.perf_counter_ns()
        copy_ns = sum(h.last_copy_ns for h in holders)
    finally:
        if was_enabled:
            gc.enable()

    # Touch state so the optimizer/runtime cannot treat the calls as useless.
    payload = sum(h.payload_elements for h in holders)
    wall_ns = t1 - t0
    outside_cpp_ns = wall_ns - copy_ns
    return {
        "mode": mode,
        "wall_ns": wall_ns,
        "cpp_copy_ns": copy_ns,
        "outside_cpp_ns": outside_cpp_ns,
        "payload": payload,
    }


def summarize(results, n):
    wall = [r["wall_ns"] / 1e6 for r in results]
    cpp = [r["cpp_copy_ns"] / 1e6 for r in results]
    outside = [r["outside_cpp_ns"] / 1e6 for r in results]

    return {
        "wall_mean": statistics.mean(wall),
        "wall_median": statistics.median(wall),
        "cpp_mean": statistics.mean(cpp),
        "outside_mean": statistics.mean(outside),
        "per_setter_ms": statistics.mean(wall) / n,
        "outside_per_setter_ms": statistics.mean(outside) / n,
    }


def main():
    p = argparse.ArgumentParser(
        description="Benchmark pybind11 conversion + C++ CDF copy costs"
    )
    p.add_argument("file")
    p.add_argument("nx", type=int)
    p.add_argument("ny", type=int)
    p.add_argument("nz", type=int)
    p.add_argument("copies", type=int)
    p.add_argument("--repeats", type=int, default=5)
    args = p.parse_args()

    expected = args.nx * args.ny * args.nz
    raw = np.fromfile(args.file, dtype=np.float32)
    if raw.size != expected:
        raise SystemExit(
            f"file has {raw.size} float32 values; expected {expected}"
        )
    image = raw.reshape(args.nz, args.ny, args.nx)

    print(f"image             : {args.nx} x {args.ny} x {args.nz}")
    print(f"voxels            : {expected:,}")
    print(f"copies/run        : {args.copies}")
    print(f"repeats           : {args.repeats}\n")

    t0 = time.perf_counter_ns()
    cdf_x, cdf_y, cdf_z = compute_image_3D_CDF(image)
    t1 = time.perf_counter_ns()
    print(f"Python CDF build  : {(t1-t0)/1e6:.3f} ms")

    (flat_x, flat_y, flat_z), flatten_ns = to_flat_arrays(cdf_x, cdf_y, cdf_z)
    print(f"CDF flatten once  : {flatten_ns/1e6:.3f} ms")
    print(f"flat sizes        : X={flat_x.size:,}, Y={flat_y.size:,}, Z={flat_z.size:,}\n")

    nested_args = (cdf_z, cdf_y, cdf_x)
    flat_args = (flat_z, flat_y, flat_x)

    modes = [
        ("nested STL caster", NestedCDFHolder, nested_args),
        ("flat STL caster", FlatSTLHolder, flat_args),
        ("flat NumPy", FlatNumpyHolder, flat_args),
    ]

    all_summaries = {}
    for name, cls, call_args in modes:
        results = []
        for _ in range(args.repeats):
            results.append(run_once(cls, call_args, args.copies, name))
        s = summarize(results, args.copies)
        all_summaries[name] = s

        print(f"=== {name} ===")
        print(f"wall mean         : {s['wall_mean']:.3f} ms total")
        print(f"wall median       : {s['wall_median']:.3f} ms total")
        print(f"wall / setter     : {s['per_setter_ms']:.3f} ms")
        print(f"C++ assignment    : {s['cpp_mean']:.3f} ms total")
        print(f"outside C++ body  : {s['outside_mean']:.3f} ms total")
        print(f"outside / setter  : {s['outside_per_setter_ms']:.3f} ms")
        print()

    nested = all_summaries["nested STL caster"]["wall_mean"]
    flat_stl = all_summaries["flat STL caster"]["wall_mean"]
    flat_np = all_summaries["flat NumPy"]["wall_mean"]

    print("=== Ratios (wall-clock) ===")
    print(f"nested / flat STL : {nested/flat_stl:.3f}x")
    print(f"nested / flat NumPy: {nested/flat_np:.3f}x")
    print(f"flat STL / NumPy  : {flat_stl/flat_np:.3f}x")


if __name__ == "__main__":
    main()

