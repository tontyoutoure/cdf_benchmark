#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

using VD  = std::vector<double>;
using VD2 = std::vector<VD>;
using VD3 = std::vector<std::vector<VD>>;

struct Dims {
    std::size_t nx = 0;
    std::size_t ny = 0;
    std::size_t nz = 0;

    std::size_t voxel_count() const {
        if (nx == 0 || ny == 0 || nz == 0)
            throw std::runtime_error("All dimensions must be > 0.");

        if (nx > std::numeric_limits<std::size_t>::max() / ny)
            throw std::runtime_error("nx*ny overflow.");
        const std::size_t nxy = nx * ny;

        if (nxy > std::numeric_limits<std::size_t>::max() / nz)
            throw std::runtime_error("nx*ny*nz overflow.");

        return nxy * nz;
    }
};

struct FlatCDF {
    std::vector<double> x; // [z][y][x]
    std::vector<double> y; // [z][y]
    std::vector<double> z; // [z]
};

struct NestedCDF {
    VD3 x; // x[z][y][x]
    VD2 y; // y[z][y]
    VD  z; // z[z]
};

static std::size_t parse_size_t(const char* s, const char* name) {
    try {
        std::string str(s);
        std::size_t pos = 0;
        unsigned long long v = std::stoull(str, &pos);
        if (pos != str.size() || v == 0 ||
            v > std::numeric_limits<std::size_t>::max())
            throw std::runtime_error("");
        return static_cast<std::size_t>(v);
    } catch (...) {
        throw std::runtime_error(std::string("Invalid ") + name + ": " + s);
    }
}

static std::vector<double> generate_weights(const Dims& d) {
    constexpr std::uint64_t seed = 0xD1B54A32D192ED03ULL;
    std::mt19937_64 rng(seed);

    std::vector<double> weights(d.voxel_count());
    for (double& weight : weights)
        weight = std::generate_canonical<double,
            std::numeric_limits<double>::digits>(rng);

    return weights;
}

static double validate_and_total(const std::vector<double>& w) {
    long double total = 0.0L;

    for (std::size_t i = 0; i < w.size(); ++i) {
        if (!std::isfinite(w[i]))
            throw std::runtime_error("NaN/Inf at flat index " + std::to_string(i));

        if (w[i] < 0.0)
            throw std::runtime_error("Negative weight at flat index " +
                                     std::to_string(i));

        total += static_cast<long double>(w[i]);
    }

    if (!(total > 0.0L))
        throw std::runtime_error("Total weight must be > 0.");

    return static_cast<double>(total);
}

template<class URBG>
static inline double uniform01(URBG& rng) {
    return std::generate_canonical<double,
        std::numeric_limits<double>::digits>(rng);
}

static FlatCDF build_flat_cdf(const std::vector<double>& w,
                              const Dims& d,
                              double total) {
    FlatCDF out;
    out.x.resize(w.size());
    out.y.resize(d.nz * d.ny);
    out.z.resize(d.nz);

    std::vector<double> row_sum(d.nz * d.ny, 0.0);
    std::vector<double> slice_sum(d.nz, 0.0);

    for (std::size_t z = 0; z < d.nz; ++z) {
        double zs = 0.0;

        for (std::size_t y = 0; y < d.ny; ++y) {
            const std::size_t row = z * d.ny + y;
            const std::size_t base = row * d.nx;

            double rs = 0.0;
            for (std::size_t x = 0; x < d.nx; ++x)
                rs += w[base + x];

            row_sum[row] = rs;
            zs += rs;
        }

        slice_sum[z] = zs;
    }

    // Z
    double accz = 0.0;
    for (std::size_t z = 0; z < d.nz; ++z) {
        accz += slice_sum[z];
        out.z[z] = accz / total;
    }
    out.z.back() = 1.0;

    // Y | Z
    for (std::size_t z = 0; z < d.nz; ++z) {
        double accy = 0.0;
        const double denom = slice_sum[z];

        for (std::size_t y = 0; y < d.ny; ++y) {
            const std::size_t row = z * d.ny + y;
            accy += row_sum[row];
            out.y[row] = denom > 0.0 ? accy / denom : 0.0;
        }

        if (denom > 0.0)
            out.y[z * d.ny + d.ny - 1] = 1.0;
    }

    // X | Z,Y
    for (std::size_t z = 0; z < d.nz; ++z) {
        for (std::size_t y = 0; y < d.ny; ++y) {
            const std::size_t row = z * d.ny + y;
            const std::size_t base = row * d.nx;
            const double denom = row_sum[row];

            double accx = 0.0;
            for (std::size_t x = 0; x < d.nx; ++x) {
                accx += w[base + x];
                out.x[base + x] = denom > 0.0 ? accx / denom : 0.0;
            }

            if (denom > 0.0)
                out.x[base + d.nx - 1] = 1.0;
        }
    }

    return out;
}

static NestedCDF build_nested_from_flat(const FlatCDF& f,
                                        const Dims& d) {
    NestedCDF out;

    out.z = f.z;

    out.y.resize(d.nz);
    for (std::size_t z = 0; z < d.nz; ++z) {
        out.y[z].resize(d.ny);
        const std::size_t base = z * d.ny;
        std::copy(f.y.begin() + static_cast<std::ptrdiff_t>(base),
                  f.y.begin() + static_cast<std::ptrdiff_t>(base + d.ny),
                  out.y[z].begin());
    }

    out.x.resize(d.nz);
    for (std::size_t z = 0; z < d.nz; ++z) {
        out.x[z].resize(d.ny);

        for (std::size_t y = 0; y < d.ny; ++y) {
            out.x[z][y].resize(d.nx);

            const std::size_t base = (z * d.ny + y) * d.nx;
            std::copy(f.x.begin() + static_cast<std::ptrdiff_t>(base),
                      f.x.begin() + static_cast<std::ptrdiff_t>(base + d.nx),
                      out.x[z][y].begin());
        }
    }

    return out;
}

template<class URBG>
static std::uint64_t sample_flat(const FlatCDF& c,
                                 const Dims& d,
                                 std::uint64_t samples,
                                 URBG& rng) {
    std::uint64_t checksum = 0;

    for (std::uint64_t s = 0; s < samples; ++s) {
        // Z
        const double pz = uniform01(rng);
        const auto zit = std::lower_bound(c.z.begin(), c.z.end(), pz);
        const std::size_t z =
            static_cast<std::size_t>(std::distance(c.z.begin(), zit));

        // Y | Z
        const double py = uniform01(rng);
        const std::size_t ybase = z * d.ny;
        const auto yb = c.y.begin() + static_cast<std::ptrdiff_t>(ybase);
        const auto ye = yb + static_cast<std::ptrdiff_t>(d.ny);
        const auto yit = std::lower_bound(yb, ye, py);
        const std::size_t y =
            static_cast<std::size_t>(std::distance(yb, yit));

        // X | Z,Y
        const double px = uniform01(rng);
        const std::size_t xbase = (z * d.ny + y) * d.nx;
        const auto xb = c.x.begin() + static_cast<std::ptrdiff_t>(xbase);
        const auto xe = xb + static_cast<std::ptrdiff_t>(d.nx);
        const auto xit = std::lower_bound(xb, xe, px);
        const std::size_t x =
            static_cast<std::size_t>(std::distance(xb, xit));

        checksum += static_cast<std::uint64_t>(
            x + 1315423911ULL * y + 2654435761ULL * z);
    }

    return checksum;
}

template<class URBG>
static std::uint64_t sample_nested(const NestedCDF& c,
                                   const Dims& d,
                                   std::uint64_t samples,
                                   URBG& rng) {
    std::uint64_t checksum = 0;

    for (std::uint64_t s = 0; s < samples; ++s) {
        // Mirrors the old implementation's access pattern.

        // Z
        const double pz = uniform01(rng);
        const auto zit = std::lower_bound(c.z.begin(), c.z.end(), pz);
        const std::size_t z =
            static_cast<std::size_t>(std::distance(c.z.begin(), zit));

        // Y | Z
        const double py = uniform01(rng);
        const auto& cy = c.y[z];
        const auto yit = std::lower_bound(cy.begin(), cy.end(), py);
        const std::size_t y =
            static_cast<std::size_t>(std::distance(cy.begin(), yit));

        // X | Z,Y
        const double px = uniform01(rng);
        const auto& cx = c.x[z][y];
        const auto xit = std::lower_bound(cx.begin(), cx.end(), px);
        const std::size_t x =
            static_cast<std::size_t>(std::distance(cx.begin(), xit));

        checksum += static_cast<std::uint64_t>(
            x + 1315423911ULL * y + 2654435761ULL * z);
    }

    return checksum;
}

static double ms(Clock::duration d) {
    return std::chrono::duration<double, std::milli>(d).count();
}

static std::size_t flat_payload_bytes(const FlatCDF& c) {
    return (c.x.size() + c.y.size() + c.z.size()) * sizeof(double);
}

static std::size_t nested_payload_bytes(const NestedCDF& c) {
    std::size_t bytes = c.z.size() * sizeof(double);

    for (const auto& row : c.y)
        bytes += row.size() * sizeof(double);

    for (const auto& plane : c.x)
        for (const auto& row : plane)
            bytes += row.size() * sizeof(double);

    return bytes;
}

static std::size_t nested_vector_object_overhead(const NestedCDF& c) {
    // Approximate object-only overhead from std::vector objects themselves.
    // Does not include allocator metadata/alignment overhead.
    std::size_t bytes = sizeof(c.z) + sizeof(c.y) + sizeof(c.x);

    bytes += c.y.size() * sizeof(VD);
    bytes += c.x.size() * sizeof(std::vector<VD>);

    for (const auto& plane : c.x)
        bytes += plane.size() * sizeof(VD);

    return bytes;
}

static void usage(const char* argv0) {
    std::cerr
        << "Usage:\n  " << argv0
        << " <nx> <ny> <nz> <num_samples> [repeats]\n\n"
        << "The program generates a reproducible random float64 array with\n"
        << "X fastest: flat = (z * ny + y) * nx + x.\n\n"
        << "repeats defaults to 5.\n";
}

int main(int argc, char** argv) {
    try {
        if (argc != 5 && argc != 6) {
            usage(argv[0]);
            return 1;
        }

        const Dims d{
            parse_size_t(argv[1], "nx"),
            parse_size_t(argv[2], "ny"),
            parse_size_t(argv[3], "nz")
        };

        const std::uint64_t samples =
            static_cast<std::uint64_t>(parse_size_t(argv[4], "num_samples"));

        const std::size_t repeats =
            argc == 6 ? parse_size_t(argv[5], "repeats") : 5;

        const std::size_t n = d.voxel_count();

        std::cout
            << "Dimensions       : " << d.nx << " x " << d.ny << " x " << d.nz << "\n"
            << "Voxel count      : " << n << "\n"
            << "Samples/run      : " << samples << "\n"
            << "Repeats          : " << repeats << "\n\n";

        const auto l0 = Clock::now();
        const std::vector<double> weights = generate_weights(d);
        const auto l1 = Clock::now();

        const double total = validate_and_total(weights);

        const auto f0 = Clock::now();
        FlatCDF flat = build_flat_cdf(weights, d, total);
        const auto f1 = Clock::now();

        const auto n0 = Clock::now();
        NestedCDF nested = build_nested_from_flat(flat, d);
        const auto n1 = Clock::now();

        std::cout << std::fixed << std::setprecision(3)
                  << "Generate time    : " << ms(l1 - l0) << " ms\n"
                  << "Flat CDF build   : " << ms(f1 - f0) << " ms\n"
                  << "Nested conversion: " << ms(n1 - n0) << " ms\n\n";

        constexpr std::uint64_t seed = 0x123456789ABCDEF0ULL;

        std::vector<double> flat_times;
        std::vector<double> nested_times;
        flat_times.reserve(repeats);
        nested_times.reserve(repeats);

        std::uint64_t checksum_flat = 0;
        std::uint64_t checksum_nested = 0;

        // Alternate execution order every repeat to reduce order bias.
        for (std::size_t r = 0; r < repeats; ++r) {
            if ((r & 1U) == 0U) {
                std::mt19937_64 rng_f(seed + r);
                const auto a0 = Clock::now();
                checksum_flat ^= sample_flat(flat, d, samples, rng_f);
                const auto a1 = Clock::now();
                flat_times.push_back(ms(a1 - a0));

                std::mt19937_64 rng_n(seed + r);
                const auto b0 = Clock::now();
                checksum_nested ^= sample_nested(nested, d, samples, rng_n);
                const auto b1 = Clock::now();
                nested_times.push_back(ms(b1 - b0));
            } else {
                std::mt19937_64 rng_n(seed + r);
                const auto b0 = Clock::now();
                checksum_nested ^= sample_nested(nested, d, samples, rng_n);
                const auto b1 = Clock::now();
                nested_times.push_back(ms(b1 - b0));

                std::mt19937_64 rng_f(seed + r);
                const auto a0 = Clock::now();
                checksum_flat ^= sample_flat(flat, d, samples, rng_f);
                const auto a1 = Clock::now();
                flat_times.push_back(ms(a1 - a0));
            }
        }

        auto summarize = [](std::vector<double> v) {
            std::sort(v.begin(), v.end());

            double sum = 0.0;
            for (double x : v) sum += x;

            const double mean = sum / static_cast<double>(v.size());
            const double median =
                (v.size() & 1U)
                    ? v[v.size() / 2]
                    : 0.5 * (v[v.size()/2 - 1] + v[v.size()/2]);

            return std::pair<double,double>(mean, median);
        };

        const auto sf = summarize(flat_times);
        const auto sn = summarize(nested_times);

        const double flat_mean_ns =
            sf.first * 1e6 / static_cast<double>(samples);
        const double nested_mean_ns =
            sn.first * 1e6 / static_cast<double>(samples);

        const double flat_median_ns =
            sf.second * 1e6 / static_cast<double>(samples);
        const double nested_median_ns =
            sn.second * 1e6 / static_cast<double>(samples);

        const double mib = 1024.0 * 1024.0;

        std::cout
            << "=== Sampling time ===\n"
            << "Flat mean        : " << sf.first << " ms"
            << "  (" << flat_mean_ns << " ns/sample)\n"
            << "Nested mean      : " << sn.first << " ms"
            << "  (" << nested_mean_ns << " ns/sample)\n"
            << "Nested/Flat mean : " << sn.first / sf.first << "x\n\n"
            << "Flat median      : " << sf.second << " ms"
            << "  (" << flat_median_ns << " ns/sample)\n"
            << "Nested median    : " << sn.second << " ms"
            << "  (" << nested_median_ns << " ns/sample)\n"
            << "Nested/Flat med. : " << sn.second / sf.second << "x\n\n";

        std::cout
            << "=== CDF memory ===\n"
            << "Flat payload     : "
            << flat_payload_bytes(flat) / mib << " MiB\n"
            << "Nested payload   : "
            << nested_payload_bytes(nested) / mib << " MiB\n"
            << "Nested vector-object overhead (approx): "
            << nested_vector_object_overhead(nested) / mib << " MiB\n\n";

        std::cout
            << "Checksums        : flat=" << checksum_flat
            << ", nested=" << checksum_nested << "\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 2;
    }
}
