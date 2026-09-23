#pragma once

#include <hnsw/detail/Node.hpp>
#include <hnsw/detail/Metric.hpp>
#include <hnsw/detail/HnswConfig.hpp>

#include <vector>
#include <optional>
#include <random>

class HnswIndexTest;

/**
 * @brief Hierarchical Navigable Small World (HNSW) index.
 * 
 * Stores vectors in a multi-layered graph and
 * allows for efficient approximate nearest neighbor search.
 */
namespace hnsw {
    template <Metric M>
    class HnswIndex {
        friend class ::HnswIndexTest;
        public:
            /** 
             * @brief Constructs an empty HNSW index
             * 
             * @param config The configuration parameters for the HNSW index.
             */
            explicit HnswIndex(
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

            std::optional<std::size_t> global_entry_point_;
            double level_multiplier_;
            std::size_t max_level_ = 0;

            std::vector<Node> nodes_;
            std::mt19937 rng_;

            mutable std::vector<uint32_t> visited_;
            mutable uint32_t generation_tag_;

            void mark_visited(std::size_t index) const noexcept {
                visited_[index] = generation_tag_;
            }

            bool is_visited(std::size_t index) const noexcept {
                return visited_[index] == generation_tag_;
            }
            
            /**
             * @brief Samples a random layer for a newly inserted node.
             *
             * @return The randomly sampled maximum layer.
             */
            std::size_t random_layer();

            /**
             * @brief Searches for nearest nodes at a specified level.
             * 
             * @param node The query node.
             * @param entry_point The entry_point for that level.
             * @param level The graph level to search.
             */
            [[nodiscard]] 
            std::vector<std::size_t> search_layer(
                const Node& node,
                std::size_t entry_point,
                std::size_t level,
                std::size_t ef
            ) const;

            /**
             * @brief Selects the best M neighbors from the 
             * given candidate and node's neighbor list.
             */
            [[nodiscard]]
            std::vector<std::size_t> select_best_neighbors(
                const Node& node,
                const std::vector<std::size_t> &candidates,
                std::size_t level
            ) const;

            /**
             * @brief Selects the best M neighbors using a heuristic from the
             * given candidate list and the node's existing neighbors.
             */
            [[nodiscard]]
            std::vector<std::size_t> select_best_neighbors_heuristic(
                const Node& node,
                const std::vector<std::size_t> &candidates,
                std::size_t level
            ) const;

            /**
             * @brief Replaces node's neighbor list with the best ones.
             * 
             * @param node The candidate node.
             * @param candidates The candidates for neighbors.
             * @param level The graph level at which the neighbors are pruned.
             */
            void prune_neighbors(
                Node& node,
                const std::vector<std::size_t> &candidates,
                std::size_t level
            );
            
            /**
             * @brief Connects the node to the specified neighbors at a given level.
             *
             * @param node The node to connect.
             * @param neighbors The neighbors to connect to.
             * @param level The graph level at which the connections are created.
             */
            void connect_neighbors(
                Node& node,
                const std::vector<std::size_t>& neighbors,
                std::size_t level
            );
    };
}

#include <hnsw/HnswIndex.tpp>