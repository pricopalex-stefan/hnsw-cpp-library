#include <hnsw/HnswBruteIndex.hpp>
#include <hnsw/EuclideanDistance.hpp>

#include <algorithm>
#include <queue>

namespace hnsw {

    void HnswBruteIndex::add(const Vector& data)
    {
        std::size_t id = nodes_.size();
        std::size_t level = 0;
        nodes_.emplace_back(id, data, level);
    }

    std::vector<std::size_t> HnswBruteIndex::search(
        const Vector& query,
        std::size_t k
    ) const
    {
        std::priority_queue<
            std::pair<float, std::size_t>,
            std::vector<std::pair<float, std::size_t>>,
            std::less<>
        > closest_neighbors;

        std::vector<std::size_t> result;

        EuclideanDistance distance_metric{};

        for(const Node& node : nodes_) {
            float dist = distance_metric(node.data(), query);
            closest_neighbors.emplace(dist, node.id());

            if(closest_neighbors.size() > k) {
                closest_neighbors.pop();
            }
        }
        while(!closest_neighbors.empty()) {
            result.push_back(closest_neighbors.top().second);
            closest_neighbors.pop();
        }

        std::ranges::reverse(result);
        return result;
    }
}