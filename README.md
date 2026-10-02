# pybind CDF conversion benchmark

This project compares several ways of transferring a three-dimensional image CDF from Python to C++. It focuses on the cost of:

- letting pybind11 convert nested Python objects to `std::vector` automatically;
- converting multidimensional NumPy arrays to `VD`, `VD2`, and `VD3` with `vector::resize` and `vector::assign`;
- copying flat NumPy arrays into `std::vector<double>` with `memcpy`;
- using `float32` input versus CDF arrays that have already been converted to and stored as `float64`.

The benchmark does not read an image file. It generates a reproducible random `float32` image with the requested dimensions.

## Project layout

- `benchmark.py` generates the random image, builds the CDF, runs the benchmarks, and reports the results.
- `src/cdfbench.cpp` contains the pybind11/C++ benchmark implementations.
- `cdfbench/__init__.py` exports the holder types provided by the native extension.
- `pyproject.toml` and `setup.py` define the Python package and native-extension build.

The nested C++ types are:

```cpp
using VD  = std::vector<double>;
using VD2 = std::vector<VD>;
using VD3 = std::vector<std::vector<VD>>;
```

The multidimensional CDF arrays use the following shapes:

```text
Z CDF: [nz]
Y CDF: [nz, ny]
X CDF: [nz, ny, nx]
```

The image and X CDF use Z-Y-X memory order, with X as the contiguous dimension.

## Requirements

- Python 3.9 or newer
- A C++ compiler with C++17 support
- NumPy
- setuptools, wheel, and pybind11 at build time

## Building with uv

From the project root, run:

```bash
uv sync
```

This creates the project environment, installs the dependencies, and builds the `cdfbench._native` extension.

After changing `src/cdfbench.cpp`, reinstall the editable package to rebuild the extension:

```bash
uv pip install --reinstall -e .
```

## Building with venv and pip

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

After changing the C++ source, rebuild it with:

```bash
python -m pip install --force-reinstall -e .
```

Verify that the native extension can be imported:

```bash
python -c "import cdfbench; print(cdfbench.__all__)"
```

If this reports `ModuleNotFoundError: cdfbench._native`, the extension has not been built, or the existing `.so` file was built for a different Python ABI. Reinstall the project from the Python environment that will run the benchmark.

## Running the benchmark

The command-line syntax is:

```bash
python benchmark.py NX NY NZ COPIES [--repeats N] [--seed N]
```

Arguments:

- `NX NY NZ` specify the X, Y, and Z image dimensions. The internal NumPy shape is `(NZ, NY, NX)`.
- `COPIES` is the number of holders created per run and therefore the number of `set_cdf` calls per run.
- `--repeats` controls the number of complete timing runs. The default is `5`.
- `--seed` controls random image generation. The default is `12345`. The same dimensions and seed produce the same input.

Example:

```bash
uv run python benchmark.py 128 128 64 10 --repeats 5 --seed 12345
```

Start with a small image and a small `COPIES` value. Every holder stores its own CDF, so memory usage grows with both the image dimensions and the number of copies.

## Implementations under test

### `nested STL caster`

This passes the nested Python CDF directly to a C++ function whose parameters are `VD`, `VD2`, and `VD3`. Before the function is entered, pybind11 recursively traverses the Python objects and constructs temporary `std::vector` instances. The C++ function then assigns those temporary vectors to the holder members.

This represents the original nested STL automatic-conversion path.

### `NumPy -> VD/VD2/VD3 (float32)`

This path accepts only C-contiguous `float32` NumPy arrays. The C++ code constructs the nested vectors as follows:

- `VD`: populate it directly with `assign`;
- `VD2`: call `resize(nz)`, then populate every row with `assign`;
- `VD3`: call `resize(nz)`, call `resize(ny)` for each Z entry, then populate every row with `assign`.

The destination vectors still contain `double`, so `assign` also converts each input value from `float` to `double`.

### `NumPy -> VD/VD2/VD3 (float64)`

This follows the same nested-vector construction path as the float32 version, but receives C-contiguous `float64` arrays. No float-to-double numeric conversion is required.

Both multidimensional NumPy interfaces use pybind11 `noconvert()` arguments. They do not silently convert an input with the wrong dtype. A dtype or shape mismatch raises an error.

### `flat NumPy (float32 input)`

This passes flat `float32` arrays to a C++ interface that requires double arrays. Before entering the C++ function, pybind11 creates temporary contiguous `float64` arrays. The C++ function then copies them into flat `std::vector<double>` members with `resize + memcpy`.

Its wall-clock time therefore includes the float32-to-float64 conversion performed before the C++ function body. The reported `C++ assignment` time contains only the subsequent `resize + memcpy` work.

### `flat NumPy (memcpy, saved float64)`

This path receives flat, contiguous CDF arrays that were converted to and stored as `float64` before the benchmark. The C++ function can use `resize + memcpy` directly, with no dtype conversion at call time.

This is the common performance baseline and is always reported as `1.000x`.

## Reading the results

### CDF preparation

The first section reports one-time preparation costs:

```text
Python CDF build  : ... ms
CDF flatten once  : ... ms (float32)
CDF -> float64    : ... ms (saved before benchmark)
```

- `Python CDF build` is the time required to build the nested three-level CDF from the random image.
- `CDF flatten once` is the time required to store the CDF as contiguous flat float32 arrays.
- `CDF -> float64` is the time required to convert and store the CDF as contiguous flat float64 arrays.

These are separate preparation measurements and are not included in the wall-clock time of any `set_cdf` benchmark. If an application converts the CDF once and calls the setter many times, the float64 conversion cost can be amortized across those calls. If conversion is required before every call, add that cost when evaluating the complete operation.

### Per-implementation timing

Each implementation reports:

```text
wall mean         : ... ms total
wall median       : ... ms total
wall / setter     : ... ms
C++ assignment    : ... ms total
outside C++ body  : ... ms total
outside / setter  : ... ms
```

- `wall mean` is the mean total wall-clock time across all repeats. Each result contains `COPIES` setter calls.
- `wall median` is the median total time. It is generally less sensitive than the mean to occasional scheduling noise.
- `wall / setter` is the mean wall-clock time divided by the number of setter calls.
- `C++ assignment` is the sum of the time measured inside the C++ function around vector assignment, `assign`, or `memcpy` operations.
- `outside C++ body` is `wall - C++ assignment`. It approximates the cost of the Python loop, pybind11 argument validation and conversion, function dispatch, and temporary-object handling outside the timed C++ body.
- `outside / setter` is that outside-body cost divided by the number of setter calls.

`outside C++ body` is the difference between two independently measured durations. It should not be treated as a precise single-component profiler result. Short runs and changes in system load can make it noisy.

### Relative performance

The final section uses this implementation as the denominator for every ratio:

```text
flat NumPy (memcpy, saved float64) = 1.000x
```

For example:

```text
NumPy -> VD/VD2/VD3 (float64)      : 1.500x
flat NumPy (memcpy, saved float64) : 1.000x
```

This means that converting multidimensional float64 NumPy arrays to nested vectors took approximately 1.5 times the baseline wall-clock time.

- A value greater than `1.000x` is slower than the baseline.
- A value equal to `1.000x` matches the baseline.
- A value less than `1.000x` is faster than the baseline.

The ratios use `wall mean`, so they include both the work inside the C++ function and all measured call overhead outside it.

## Tips for stable measurements

- Use an optimized build. `setup.py` currently enables `-O3`.
- Keep `--seed` fixed so every implementation processes identical data.
- Increase `--repeats` and `COPIES` until each run is long enough to dominate timer and function-call noise.
- Avoid running other CPU- or memory-intensive workloads during the benchmark.
- Inspect the mean, median, and per-setter values instead of relying on a single run.
- When comparing different systems, record the compiler, Python, NumPy, and pybind11 versions.
