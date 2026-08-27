#pragma once

#include <hnsw/Vector.hpp>
#include <cmath>

namespace hnsw {
    class EuclideanDistance final {
        public:
            [[nodiscard]]
            float operator()(
                const Vector& u,
                const Vector& v
            ) const noexcept
            {
                float sum = 0.0f;
                for(std::size_t i = 0; i < u.size(); i++) {
                    const float diff = u[i] - v[i];
                    sum += diff * diff;
                }
                return sqrt(sum);
            }
    };
}