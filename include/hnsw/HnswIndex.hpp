#pragma once

#include <hnsw/Node.hpp>
#include <hnsw/Metric.hpp>
#include <hnsw/HnswConfig.hpp>

#include <vector>
#include <optional>

/**
 * @brief Hierarchical Navigable Small World (HNSW) index.
 * 
 * Stores vectors in a multi-layered graph and
 * allows for efficient approximate nearest neighbor search.
 */
namespace hnsw {
    template <Metric M>
    class HnswIndex {
        public:
            /** 
             * @brief Constructs an empty HNSW index
             * 
             * @param metric The distance metric to use for comparing vectors.
             * @param config The configuration parameters for the HNSW index.
             */
            HnswIndex(
                M metric,
                HnswConfig config = {}
            );

            /** 
             * @brief Adds a vector to the HNSW index.
             * 
             * @param data The vector to add.
             */
            void add(const Vector& data);

            /** 
             * @brief Adds multiple vectors to the HNSW index.
             * 
             * @param data The vectors to add.
             */
            void add(const std::vector<Vector>& data);

            /**
             * @brief Searches for the k nearest neighbors of a query vector.
             * 
             * @param query The query vector.
             * @param k The number of nearest neighbors to return.
             * @return A vector of indices of the k nearest neighbors in the index.
             */
            [[nodiscard]]
            std::vector<std::size_t> search(
                const Vector& query,
                std::size_t k
            ) const;
        
        private:
            M metric_;
            HnswConfig config_;

            std::optional<std::size_t> entry_point_;
            std::size_t max_level_ = 0;

            std::vector<Node> nodes_;
    };
}