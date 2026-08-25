#pragma once

#include <hnsw/Vector.hpp>
#include <concepts>

namespace hnsw {
    template <typename M>
    concept Metric = requires(
        M metric,
        const Vector& u,
        const Vector& v
    ) 
    {
        { metric(u, v) } noexcept -> std::same_as<float>;
    };
}