#pragma once

namespace hnsw {
    struct HnswConfig {
        std::size_t M = 16;
        std::size_t ef_construction = 24;
        std::size_t ef_search = 16;
    };
}