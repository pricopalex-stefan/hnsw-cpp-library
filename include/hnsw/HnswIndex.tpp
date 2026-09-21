#include <utility>
#include <queue>
#include <algorithm>
#include <cstddef>

namespace hnsw {

    template <Metric M>
    HnswIndex<M>::HnswIndex(
        HnswConfig config)
    : config_(config),
      metric_{},
      rng_(std::random_device{}()),
      level_multiplier_(1.0 / std::log(1.0 * config_.M))
    {
        std::ranges::fill(visited_, 0);
        generation_tag_ = 1;
    }

    // Sample nodes highest layer by using exponential distribution
    template <Metric M>
    std::size_t HnswIndex<M>::random_layer()
    {
        std::uniform_real_distribution<double> dis(0.0, 1.0);
        double u = dis(this->rng_);
        return static_cast<std::size_t>(
            -std::log(u) * this->level_multiplier_
        );
    }

    template<Metric M>
    std::vector<std::size_t> HnswIndex<M>::search_layer(
        const Node& node,
        std::size_t entry_point,
        std::size_t level,
        std::size_t ef /*.ef_construction or .ef_search*/
    ) const
    {

        auto max_compare = [](std::pair<std::size_t, float> left, 
            std::pair<std::size_t, float> right) {
                return left.second > right.second;
            };

        auto min_compare = [](std::pair<std::size_t, float> left, 
            std::pair<std::size_t, float> right) {
                return left.second < right.second;
            };

        std::priority_queue<
            std::pair<std::size_t, float>,
            std::vector<std::pair<std::size_t, float>>, 
            decltype(max_compare)
        > candidates(max_compare);

        std::priority_queue<
            std::pair<std::size_t, float>, 
            std::vector<std::pair<std::size_t, float>>,
            decltype(min_compare)
        > result(min_compare);

        candidates.emplace(entry_point, metric_(node.data(), nodes_[entry_point].data()));
        result.emplace(entry_point, metric_(node.data(), nodes_[entry_point].data()));

        generation_tag_++;

        // Overflow
        if(generation_tag_ == 0) {
            std::ranges::fill(visited_, 0);
            generation_tag_ = 1;
        }

        /**
         * Explore the candidate neighbors and keep at most ef results.
         * Candidates are explored from closest to farthest, while the result keeps
         * the farthest current result at its top and the candidate heap keeps the closest
         * on the top. Exploration stops when the closest candidate cannot improve
         * the result list. 
        */

        while(!candidates.empty() && !result.empty()) {

            const float best_candidate_distance = candidates.top().second;
            const float worst_result_distance = result.top().second;
            if(result.size() >= ef &&
                best_candidate_distance > worst_result_distance) {break;}

            std::size_t candidate_id = candidates.top().first;
            candidates.pop();
            for(const std::size_t neighbor_id : nodes_[candidate_id].neighbors(level)) {
                // Process each node only once.
                if(!is_visited(neighbor_id)) {
                    mark_visited(neighbor_id);
                    candidates.emplace(neighbor_id, metric_(node.data(), nodes_[neighbor_id].data()));
                    if(result.size() < ef) {
                        result.emplace(neighbor_id, metric_(node.data(), nodes_[neighbor_id].data()));
                    } else if(metric_(node.data(), nodes_[neighbor_id].data())
                        < result.top().second) {
                        result.pop();
                        result.emplace(neighbor_id, metric_(node.data(), nodes_[neighbor_id].data()));
                    }
                }
            }
        }

        std::vector<std::size_t> result_vector;
        while(!result.empty()) {
            result_vector.push_back(result.top().first);
            result.pop();
        }
        std::ranges::reverse(result_vector);
        return result_vector;
    }

    /**
     * Select best neighbors by storing all current neighbors 
     * and candidates in max heap and selecting a maximum of M.
     */
    template<Metric M>
    std::vector<std::size_t> HnswIndex<M>::select_best_neighbors(
        const Node& node,
        const std::vector<std::size_t> &candidates,
        std::size_t level
    ) const
    {
            auto max_compare = [](std::pair<std::size_t, float> left, 
                std::pair<std::size_t, float> right) {
                    return left.second > right.second;
                };

        std::priority_queue<
            std::pair<std::size_t, float>,
            std::vector<std::pair<std::size_t, float>>,                
            decltype(max_compare)
        > closest_neighbors(max_compare);

        for(const std::size_t candidate_id : candidates) {
            closest_neighbors.emplace(candidate_id, metric_(node.data(), nodes_[candidate_id].data()));

            if(closest_neighbors.size() > config_.M) {
                closest_neighbors.pop();
            }
        }

        for(const std::size_t neighbor_id : node.neighbors(level)) {
            closest_neighbors.emplace(neighbor_id, metric_(node.data(), nodes_[neighbor_id].data()));

            if(closest_neighbors.size() > config_.M) {
                closest_neighbors.pop();
            }
        }

        std::vector<std::size_t> selected;
        while(!closest_neighbors.empty()) {
            selected.push_back(closest_neighbors.top().first);
            closest_neighbors.pop();
        }

        return selected;
    }

    /**
    * Select best neighbors that are closer to 
    * node than any node in the result that 
    * favours geometric diversity creating 
    * more global connections.
    */
    template<Metric M>
    std::vector<std::size_t> HnswIndex<M>::select_best_neighbors_heuristic(
        const Node& node,
        const std::vector<std::size_t> &candidates,
        std::size_t level
    ) const 
    {
        auto max_compare = [](std::pair<std::size_t, float> left, 
            std::pair<std::size_t, float> right) {
                return left.second > right.second;
            };

        std::priority_queue<
            std::pair<std::size_t, float>, 
            std::vector<std::pair<std::size_t, float>>,                
            decltype(max_compare)
        > candidates_queue(max_compare);

        std::priority_queue<
            std::pair<std::size_t, float>, 
            std::vector<std::pair<std::size_t, float>>,                
            decltype(max_compare)
        > candidates_discarded(max_compare);

        std::vector<std::size_t> result;


        for(std::size_t neighbors_id : node.neighbors(level)) {
            candidates_queue.emplace(neighbors_id, metric_(node.data(), nodes_[neighbors_id].data()));
        }

        for(std::size_t candidates_id : candidates) {
            candidates_queue.emplace(candidates_id, metric_(node.data(), nodes_[candidates_id].data()));
        }

        while(!candidates_queue.empty() && result.size() < config_.M) {
            const std::size_t best_candidate_id = candidates_queue.top().first;
            const float distance_to_q = candidates_queue.top().second;
            candidates_queue.pop();

            bool add_to_result = true;

            for(std::size_t result_node_id : result) {
                const float distance_to_r = metric_(nodes_[best_candidate_id].data(), nodes_[result_node_id].data());
                // Reject the candidate if it is closer to an already chosen node
                // than it is to the query node
                if(distance_to_q >= distance_to_r) {
                    add_to_result = false;
                    candidates_discarded.emplace(best_candidate_id, distance_to_q);
                    break;
                }
            }

            if(add_to_result) {
                result.push_back(best_candidate_id);
            }
        }

        // Add the remaining candidates nodes to result
        while(!candidates_discarded.empty() && result.size() < config_.M) {
            result.push_back(candidates_discarded.top().first);
            candidates_discarded.pop();
        }

        return result;
    }

    template<Metric M>
    void HnswIndex<M>::prune_neighbors(
        Node& node,
        const std::vector<std::size_t> &candidates,
        std::size_t level
    )
    {
        std::vector<std::size_t> selected = 
                select_best_neighbors_heuristic(node, candidates, level);

        node.replace_neighbors(selected, level);
    }

    template <Metric M>
    void HnswIndex<M>::connect_neighbors(
        Node& node,
        const std::vector<std::size_t>& neighbors,
        std::size_t level
    )
    {
        prune_neighbors(node, neighbors, level);

        // Bidirectional linking , every node keeping at most M nodes

        for(const auto neighbor_id : node.neighbors(level)) {

            Node& neighbor_node = nodes_[neighbor_id];

            neighbor_node.add_neighbor(node.id(), level);

            prune_neighbors(neighbor_node, std::vector<std::size_t>{node.id()}, level);
        }
    }

    template <Metric M>
    void HnswIndex<M>::add(const Vector& data) {
        
        std::size_t level_to_insert = random_layer();
        Node nodeToInsert(nodes_.size(), data, level_to_insert);

        /** 
         * If the index is empty, insert the first node and use it 
         * as the global entry point
        */
        if(!global_entry_point_) {
            nodes_.push_back(nodeToInsert);
            visited_.push_back(0);
            global_entry_point_ = nodeToInsert.id();
            max_level_ = level_to_insert;
        } else {
            nodes_.push_back(std::move(nodeToInsert));
            visited_.push_back(0);
            Node& node = nodes_.back();

            std::size_t current_entry_point = *global_entry_point_;

            const std::size_t start_level = std::min(level_to_insert, max_level_);

            // Perform greedy navigation through the upper levels.
            for(std::size_t l = max_level_; l > level_to_insert; --l) {

                float min_dist = metric_(nodes_[current_entry_point].data(), node.data());
        
                for(std::size_t neighbor_id : nodes_[current_entry_point].neighbors(l)) {

                    const float dist_neighbor = metric_(nodes_[neighbor_id].data(), node.data());
                    if(dist_neighbor < min_dist) {
                        min_dist = dist_neighbor;
                        current_entry_point = neighbor_id;
                    }
                }
            }

            // Search for candidate neighbors and connect to the node
            // at each level down to level 0
            for(std::size_t level = start_level;; --level) {
                std::vector<std::size_t> closest_neighbors = search_layer(node, current_entry_point, level, config_.ef_construction);
                connect_neighbors(node, closest_neighbors, level);

                if(level == 0) {
                    break;
                }
            }

            // If the new node has the highest level , make it the new global entry point
            if(max_level_ < level_to_insert) {
                max_level_ = level_to_insert;
                global_entry_point_ = node.id();
            }
        }
    }

    template<Metric M> 
    void HnswIndex<M>::add(const std::vector<Vector>& data) {
        for(const Vector& vector : data) {
            this->add(vector);
        }
    }

    template<Metric M>
    std::vector<std::size_t> HnswIndex<M>::search(
                const Vector& query,
                std::size_t k
    ) const
    {
        if(!global_entry_point_.has_value() || k == 0) {
            return {};
        }
        std::vector<std::size_t> candidates;
        std::size_t current_entry_point = *global_entry_point_;
        Node query_node(
            nodes_.size(),
            query,
            0
        );

        // Perform greedy search until level 0.
        for(std::size_t level = max_level_; level > 0; --level) {

            while(true) {
                const float min_dist =
                    metric_(nodes_[current_entry_point].data(), query);

                std::size_t best_neighbor = current_entry_point;
                float best_dist = min_dist;

                for(const std::size_t neighbor_id :
                    nodes_[current_entry_point].neighbors(level)) {

                    const float neighbor_dist =
                        metric_(
                            nodes_[neighbor_id].data(),
                            query
                        );

                    if(neighbor_dist < best_dist) {
                        best_neighbor = neighbor_id;
                        best_dist = neighbor_dist;
                    }
                }

                if(best_neighbor == current_entry_point) {
                    break;
                }

                current_entry_point = best_neighbor;
            }
        }

        // Perform layer search at layer 0.
        candidates = search_layer(
            query_node,
            current_entry_point,
            0,
            config_.ef_search
        );

        const std::size_t result_count = std::min(candidates.size(), k);

        const auto candidates_end = 
            candidates.begin() + static_cast<std::ptrdiff_t>(result_count);

        return {
            candidates.begin(),
            candidates_end
        };
    }
}