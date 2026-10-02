#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace py = pybind11;
using Clock = std::chrono::steady_clock;

using VD  = std::vector<double>;
using VD2 = std::vector<VD>;
using VD3 = std::vector<std::vector<VD>>;

static std::uint64_t elapsed_ns(Clock::time_point a, Clock::time_point b) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count());
}

class NestedCDFHolder {
public:
    // This deliberately mirrors the old Gate signature.  pybind11 must first
    // convert Python objects to VD/VD2/VD3 before this function is entered.
    void set_cdf(const VD& vz, const VD2& vy, const VD3& vx) {
        const auto t0 = Clock::now();
        fCDFZ = vz;
        fCDFY = vy;
        fCDFX = vx;
        const auto t1 = Clock::now();
        last_copy_ns_ = elapsed_ns(t0, t1);
    }

    std::uint64_t last_copy_ns() const { return last_copy_ns_; }

    std::size_t payload_elements() const {
        std::size_t n = fCDFZ.size();
        for (const auto& v : fCDFY) n += v.size();
        for (const auto& vv : fCDFX)
            for (const auto& v : vv) n += v.size();
        return n;
    }

private:
    VD3 fCDFX;
    VD2 fCDFY;
    VD  fCDFZ;
    std::uint64_t last_copy_ns_ = 0;
};

template <typename T>
class NestedNumpyHolder {
public:
    // Convert contiguous NumPy arrays with shapes [Z], [Z,Y], [Z,Y,X]
    // directly into VD/VD2/VD3.  The outer vectors are resized and every
    // innermost vector is populated with vector::assign.
    void set_cdf(
        py::array_t<T, py::array::c_style> vz,
        py::array_t<T, py::array::c_style> vy,
        py::array_t<T, py::array::c_style> vx) {

        const auto bz = vz.request();
        const auto by = vy.request();
        const auto bx = vx.request();
        validate_shapes(bz, by, bx);

        const auto nz = static_cast<std::size_t>(bz.shape[0]);
        const auto ny = static_cast<std::size_t>(by.shape[1]);
        const auto nx = static_cast<std::size_t>(bx.shape[2]);
        const auto* pz = static_cast<const T*>(bz.ptr);
        const auto* py_data = static_cast<const T*>(by.ptr);
        const auto* px = static_cast<const T*>(bx.ptr);

        const auto t0 = Clock::now();
        fCDFZ.assign(pz, pz + nz);

        fCDFY.resize(nz);
        for (std::size_t z = 0; z < nz; ++z) {
            const T* row = py_data + z * ny;
            fCDFY[z].assign(row, row + ny);
        }

        fCDFX.resize(nz);
        for (std::size_t z = 0; z < nz; ++z) {
            fCDFX[z].resize(ny);
            for (std::size_t y = 0; y < ny; ++y) {
                const T* row = px + (z * ny + y) * nx;
                fCDFX[z][y].assign(row, row + nx);
            }
        }
        const auto t1 = Clock::now();
        last_copy_ns_ = elapsed_ns(t0, t1);
    }

    std::uint64_t last_copy_ns() const { return last_copy_ns_; }

    std::size_t payload_elements() const {
        std::size_t n = fCDFZ.size();
        for (const auto& v : fCDFY) n += v.size();
        for (const auto& vv : fCDFX)
            for (const auto& v : vv) n += v.size();
        return n;
    }

private:
    static void validate_shapes(
        const py::buffer_info& bz,
        const py::buffer_info& by,
        const py::buffer_info& bx) {
        if (bz.ndim != 1 || by.ndim != 2 || bx.ndim != 3)
            throw std::runtime_error(
                "Expected CDF shapes [Z], [Z,Y], and [Z,Y,X]");
        if (by.shape[0] != bz.shape[0] ||
            bx.shape[0] != bz.shape[0] || bx.shape[1] != by.shape[1])
            throw std::runtime_error("CDF array dimensions do not match");
    }

    VD3 fCDFX;
    VD2 fCDFY;
    VD fCDFZ;
    std::uint64_t last_copy_ns_ = 0;
};

class FlatSTLHolder {
public:
    // Flat std::vector parameters: pybind11 still converts Python iterables
    // element-by-element to std::vector<double>, but there is no nested caster.
    void set_cdf(const VD& vz, const VD& vy, const VD& vx) {
        const auto t0 = Clock::now();
        fCDFZ = vz;
        fCDFY = vy;
        fCDFX = vx;
        const auto t1 = Clock::now();
        last_copy_ns_ = elapsed_ns(t0, t1);
    }

    std::uint64_t last_copy_ns() const { return last_copy_ns_; }
    std::size_t payload_elements() const {
        return fCDFX.size() + fCDFY.size() + fCDFZ.size();
    }

private:
    VD fCDFX, fCDFY, fCDFZ;
    std::uint64_t last_copy_ns_ = 0;
};

class FlatNumpyHolder {
public:
    // Fast path: accept contiguous float64 NumPy arrays directly and memcpy
    // into three flat C++ vectors.  If the Python argument is already C-order
    // float64, pybind11 does not need to walk every element with STL casters.
    void set_cdf(
        py::array_t<double, py::array::c_style | py::array::forcecast> vz,
        py::array_t<double, py::array::c_style | py::array::forcecast> vy,
        py::array_t<double, py::array::c_style | py::array::forcecast> vx) {
        auto size_z = vz.size();
        auto size_y = vy.size();
        auto size_x = vx.size();
        const auto t0 = Clock::now();
        fCDFZ.assign(vz.data(), vz.data() + size_z);
        fCDFY.assign(vy.data(), vy.data() + size_y);
        fCDFX.assign(vx.data(), vx.data() + size_x);
        const auto t1 = Clock::now();
        last_copy_ns_ = elapsed_ns(t0, t1);
    }

    std::uint64_t last_copy_ns() const { return last_copy_ns_; }
    std::size_t payload_elements() const {
        return fCDFX.size() + fCDFY.size() + fCDFZ.size();
    }

private:
    static void copy_array(const py::buffer_info& b, VD& dst) {
        if (b.itemsize != static_cast<py::ssize_t>(sizeof(double)))
            throw std::runtime_error("Expected float64 array");

        std::size_t n = 1;
        for (auto dim : b.shape)
            n *= static_cast<std::size_t>(dim);

        dst.resize(n);
        if (n != 0)
            std::memcpy(dst.data(), b.ptr, n * sizeof(double));
    }

    VD fCDFX, fCDFY, fCDFZ;
    std::uint64_t last_copy_ns_ = 0;
};

PYBIND11_MODULE(_native, m) {
    m.doc() = "pybind11 nested-vs-flat CDF conversion benchmark";

    py::class_<NestedCDFHolder>(m, "NestedCDFHolder")
        .def(py::init<>())
        .def("set_cdf", &NestedCDFHolder::set_cdf,
             py::arg("vz"), py::arg("vy"), py::arg("vx"))
        .def_property_readonly("last_copy_ns", &NestedCDFHolder::last_copy_ns)
        .def_property_readonly("payload_elements", &NestedCDFHolder::payload_elements);

    py::class_<NestedNumpyHolder<float>>(m, "NestedNumpyF32Holder")
        .def(py::init<>())
        .def("set_cdf", &NestedNumpyHolder<float>::set_cdf,
             py::arg("vz").noconvert(),
             py::arg("vy").noconvert(),
             py::arg("vx").noconvert())
        .def_property_readonly(
            "last_copy_ns", &NestedNumpyHolder<float>::last_copy_ns)
        .def_property_readonly(
            "payload_elements", &NestedNumpyHolder<float>::payload_elements);

    py::class_<NestedNumpyHolder<double>>(m, "NestedNumpyF64Holder")
        .def(py::init<>())
        .def("set_cdf", &NestedNumpyHolder<double>::set_cdf,
             py::arg("vz").noconvert(),
             py::arg("vy").noconvert(),
             py::arg("vx").noconvert())
        .def_property_readonly(
            "last_copy_ns", &NestedNumpyHolder<double>::last_copy_ns)
        .def_property_readonly(
            "payload_elements", &NestedNumpyHolder<double>::payload_elements);

    py::class_<FlatSTLHolder>(m, "FlatSTLHolder")
        .def(py::init<>())
        .def("set_cdf", &FlatSTLHolder::set_cdf,
             py::arg("vz"), py::arg("vy"), py::arg("vx"))
        .def_property_readonly("last_copy_ns", &FlatSTLHolder::last_copy_ns)
        .def_property_readonly("payload_elements", &FlatSTLHolder::payload_elements);

    py::class_<FlatNumpyHolder>(m, "FlatNumpyHolder")
        .def(py::init<>())
        .def("set_cdf", &FlatNumpyHolder::set_cdf,
             py::arg("vz"), py::arg("vy"), py::arg("vx"))
        .def_property_readonly("last_copy_ns", &FlatNumpyHolder::last_copy_ns)
        .def_property_readonly("payload_elements", &FlatNumpyHolder::payload_elements);
}
