#pragma once

#include "Vector.hpp"

namespace hnsw {
    class DotProductDistance final {
        public:
            [[nodiscard]]
            float operator()(
                const Vector& u,
                const Vector& v
            ) const noexcept
            {
                float dot_product = 0.0f;
                for(std::size_t i = 0; i < u.size(); i++) {
                    dot_product += u[i] * v[i];
                }
                return dot_product;
            }
    };
}
