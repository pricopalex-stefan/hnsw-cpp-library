#pragma once

#include <hnsw/Vector.hpp>
#include <vector>

namespace hnsw {

    class Node {

        public:          
            Node(std::size_t id, Vector data, std::size_t level);
            [[nodiscard]] std::size_t id() const noexcept;
            [[nodiscard]] const Vector& data() const noexcept;
            [[nodiscard]] std::size_t level() const noexcept;
            [[nodiscard]] const std::vector<std::size_t>& neighbors(std::size_t level) const noexcept;
            void add_neighbor(std::size_t neighbor_id, std::size_t level) noexcept;


        private:
            std::size_t id_;
            Vector data_;
            std::size_t level_;
            std::vector<std::vector<std::size_t>> neighbors_;
    };

}