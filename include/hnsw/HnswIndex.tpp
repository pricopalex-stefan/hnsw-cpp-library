#include <utility>
#include <queue>
#include <unordered_set>
#include <algorithm>

namespace hnsw {

    template <Metric M>
    HnswIndex<M>::HnswIndex(
        HnswConfig config)
    : config_(config),
      metric_{},
      rng_(std::random_device{}()),
      level_multiplier_(1.0 / std::log(1.0 * config_.M))
    {

    }

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
        std::size_t level
    ) const
    {
        DistanceComparator comparatorMaxHeap{
            node, this->nodes_, this->metric_, true
        };
        DistanceComparator comparatorMinHeap{
            node, this->nodes_, this->metric_, false
        };

        std::priority_queue<
            std::size_t, std::vector<std::size_t>, 
            DistanceComparator
        > candidates(comparatorMinHeap);

        std::priority_queue<
            std::size_t, 
            std::vector<std::size_t>, 
            DistanceComparator
        > result(comparatorMaxHeap);
        
        std::unordered_set<std::size_t> visited;

        // Inserting initial node ids into the candidates min heap and into result max heap.
        for(const std::size_t neighbor_id : node.neighbors(level)) {
            candidates.push(neighbor_id);
            if(result.size() < config_.ef_construction) {
                result.push(neighbor_id);

            // Replace worst result if the new candidate is closer.
            } else if(metric_(node.data(), nodes_[neighbor_id].data())
                    < metric_(node.data(), nodes_[result.top()].data())){
                result.pop();
                result.push(neighbor_id);
            }
            visited.insert(neighbor_id);
        }

        /**
         * Explore the candidate neighbors and keep at most ef_construction results.
         * Candidates are explored from closest to farthest, while the result keeps
         * the farthest current result at its top and the candidate the keeps the closest
         * on the top. Exploration stops when the closest candidate cannot improve
         * the result list. 
        */

        while(!candidates.empty() && !result.empty()) {

            const float best_candidate_distance = metric_(node.data(), nodes_[candidates.top()].data());
            const float worst_result_distance = metric_(node.data(), nodes_[result.top()].data());
            if(best_candidate_distance > worst_result_distance) {break;}

            std::size_t candidate_id = candidates.top();
            candidates.pop();
            for(const std::size_t neighbor_id : nodes_[candidate_id].neighbors(level)) {
                // Process each node only once.
                if(visited.insert(neighbor_id).second) {
                    candidates.push(neighbor_id);
                    if(result.size() < config_.ef_construction) {
                        result.push(neighbor_id);
                    } else if(metric_(node.data(), nodes_[neighbor_id].data())
                        < metric_(node.data(), nodes_[result.top()].data())) {
                        result.pop();
                        result.push(neighbor_id);
                    }
                }
            }
        }

        std::vector<std::size_t> result_vector;
        while(!result.empty()) {
            result_vector.push_back(result.top());
            result.pop();
        }
        std::ranges::reverse(result_vector);
        return result_vector;
    }
}