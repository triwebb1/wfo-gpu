#pragma once
// Portable stand-ins for std::normal_distribution and
// std::uniform_real_distribution.
//
// std::mt19937 produces the same sequence everywhere, but the standard
// distributions are implementation-defined, so libstdc++ and MSVC turn the
// same engine output into different values. Tests that build synthetic
// prices from a seed use these instead, so they see identical data on every
// platform.

#include <cstdint>

namespace portable {

// Uniform in [a, b) from one 32-bit draw.
template <class T>
struct uniform {
    T a, b;
    template <class G>
    T operator()(G& g) {
        const double u = static_cast<double>(static_cast<uint32_t>(g())) / 4294967296.0;
        return static_cast<T>(a + (b - a) * u);
    }
};

// Approximately normal(mean, sd): sum of 12 uniforms (Irwin-Hall), which
// needs no transcendental functions.
template <class T>
struct normal {
    T mean, sd;
    template <class G>
    T operator()(G& g) {
        double sum = 0.0;
        for (int i = 0; i < 12; ++i)
            sum += static_cast<double>(static_cast<uint32_t>(g())) / 4294967296.0;
        return static_cast<T>(mean + sd * (sum - 6.0));
    }
};

}  // namespace portable
