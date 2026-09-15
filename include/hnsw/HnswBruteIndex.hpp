#pragma once

#include <hnsw/Vector.hpp>
#include <hnsw/Node.hpp>

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