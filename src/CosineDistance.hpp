#pragma once

#include "Vector.hpp"
#include <cmath>

namespace hnsw {
    class CosineDistance final {
        public:
            [[nodiscard]]
            float operator()(
                const Vector& u,
                const Vector& v
            ) const noexcept 
            {
                float dot_product = 0.0f;
                float magnitude_u_squared = 0.0f;
                float magnitude_v_squared = 0.0f;
                for(std::size_t i = 0; i < u.size(); i++) {
                    dot_product += u[i] * v[i];
                    magnitude_u_squared += u[i] * u[i];
                    magnitude_v_squared += v[i] * v[i];
                }

                float magnitude_u = std::sqrt(magnitude_u_squared);
                float magnitude_v = std::sqrt(magnitude_v_squared);

                float similarity = 
                    dot_product / (magnitude_u * magnitude_v);

                return 1.0f - similarity;
            }
    };
}