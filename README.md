# pybind CDF conversion benchmark

Benchmarks several ways to transfer a 3D image CDF from Python/NumPy to C++ `std::vector<double>` containers.

## Build and run

Requires Python 3.9+, NumPy, a C++17 compiler, and either uv or pip.
```bash
uv sync
uv run python benchmark.py NX NY NZ COPIES --repeats 5 --seed 12345
```
Alternatively, run `python -m pip install -e .`, then invoke `python benchmark.py` with the same arguments.

## Results

- `nested STL caster`: pybind11 recursively converts Python objects to `VD/VD2/VD3`.
- `NumPy -> VD/VD2/VD3`: strict C-contiguous f32/f64 arrays are copied with `resize` and `assign`; f32 is converted to double during assignment.
- `flat NumPy (float32 input)`: pybind11 converts to float64, then C++ uses `memcpy`.
- `flat NumPy (memcpy, saved float64)`: direct `memcpy` path and the `1.000x` baseline.
- `wall` measures the complete call; `C++ assignment` measures only C++ copying; `outside C++ body` is their difference. Preparation times such as CDF construction and float64 conversion are reported separately and excluded from setter timings.
