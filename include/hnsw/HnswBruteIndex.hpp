#pragma once

#include <hnsw/detail/Vector.hpp>
#include <hnsw/detail/Node.hpp>

namespace hnsw {
    class HnswBruteIndex {
        public:
            HnswBruteIndex() = default;

            void add(const Vector& data);

            [[nodiscard]]
            std::vector<std::size_t> search(
                const Vector& query,
                std::size_t k
            ) const;
        private:
            std::vector<Node> nodes_;
    };
} 