#pragma once

#include <hnsw/Vector.hpp>
#include <vector>

namespace hnsw {

    class Node {

        public:          
            Node(int id, Vector data, int level);
            [[nodiscard]] int id() const noexcept;
            [[nodiscard]] const Vector& data() const noexcept;
            [[nodiscard]] int level() const noexcept;


        private:
            int id_;
            Vector data_;
            int level_;
            std::vector<std::vector<std::size_t>> neighbors_;
    };

}