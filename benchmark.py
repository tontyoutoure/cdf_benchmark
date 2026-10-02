#!/usr/bin/env python3
import argparse
import gc
import statistics
import time

import numpy as np

from cdfbench import (
    FlatNumpyHolder,
    NestedCDFHolder,
    NestedNumpyF32Holder,
    NestedNumpyF64Holder,
)


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


def to_flat_arrays(cdf_x, cdf_y, cdf_z, dtype=None):
    """Flatten CDFs, optionally converting and storing them in a new dtype."""
    t0 = time.perf_counter_ns()
    x = np.ascontiguousarray(np.asarray(cdf_x, dtype=dtype)).reshape(-1)
    y = np.ascontiguousarray(np.asarray(cdf_y, dtype=dtype)).reshape(-1)
    z = np.ascontiguousarray(np.asarray(cdf_z, dtype=dtype)).reshape(-1)
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
    p.add_argument("nx", type=int)
    p.add_argument("ny", type=int)
    p.add_argument("nz", type=int)
    p.add_argument("copies", type=int)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="seed used to generate the random float32 image (default: 12345)",
    )
    args = p.parse_args()

    if min(args.nx, args.ny, args.nz, args.copies, args.repeats) <= 0:
        p.error("nx, ny, nz, copies, and repeats must all be positive")

    expected = args.nx * args.ny * args.nz
    rng = np.random.default_rng(args.seed)
    image = rng.random((args.nz, args.ny, args.nx), dtype=np.float32)

    print(f"image             : {args.nx} x {args.ny} x {args.nz}")
    print(f"voxels            : {expected:,}")
    print(f"random seed       : {args.seed}")
    print(f"copies/run        : {args.copies}")
    print(f"repeats           : {args.repeats}\n")

    t0 = time.perf_counter_ns()
    cdf_x, cdf_y, cdf_z = compute_image_3D_CDF(image)
    t1 = time.perf_counter_ns()
    print(f"Python CDF build  : {(t1-t0)/1e6:.3f} ms")

    (flat_x, flat_y, flat_z), flatten_ns = to_flat_arrays(
        cdf_x, cdf_y, cdf_z
    )
    (flat64_x, flat64_y, flat64_z), flatten64_ns = to_flat_arrays(
        cdf_x, cdf_y, cdf_z, dtype=np.float64
    )
    print(f"CDF flatten once  : {flatten_ns/1e6:.3f} ms ({flat_x.dtype})")
    print(f"CDF -> float64    : {flatten64_ns/1e6:.3f} ms (saved before benchmark)")
    print(
        f"flat sizes        : X={flat_x.size:,}, Y={flat_y.size:,}, "
        f"Z={flat_z.size:,}\n"
    )

    nested_args = (cdf_z, cdf_y, cdf_x)
    flat_args = (flat_z, flat_y, flat_x)
    flat64_args = (flat64_z, flat64_y, flat64_x)
    nested_numpy_args = (
        flat_z.reshape(args.nz),
        flat_y.reshape(args.nz, args.ny),
        flat_x.reshape(args.nz, args.ny, args.nx),
    )
    nested_numpy64_args = (
        flat64_z.reshape(args.nz),
        flat64_y.reshape(args.nz, args.ny),
        flat64_x.reshape(args.nz, args.ny, args.nx),
    )

    baseline_name = "flat NumPy (memcpy, saved float64)"
    modes = [
        ("nested STL caster", NestedCDFHolder, nested_args),
        ("NumPy -> VD/VD2/VD3 (float32)", NestedNumpyF32Holder, nested_numpy_args),
        ("NumPy -> VD/VD2/VD3 (float64)", NestedNumpyF64Holder, nested_numpy64_args),
        (f"flat NumPy ({flat_x.dtype} input)", FlatNumpyHolder, flat_args),
        (baseline_name, FlatNumpyHolder, flat64_args),
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

    baseline = all_summaries[baseline_name]["wall_mean"]
    print(f"=== Ratios (wall-clock, baseline: {baseline_name}) ===")
    label_width = max(len(name) for name, _, _ in modes)
    for name, _, _ in modes:
        wall_mean = all_summaries[name]["wall_mean"]
        print(f"{name:<{label_width}} : {wall_mean / baseline:.3f}x")


if __name__ == "__main__":
    main()
