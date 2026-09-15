#pragma once

#include <vector>
#include <unordered_set>

float recallAtK(
    const std::vector<std::size_t>& exact_neighbors,
    const std::vector<std::size_t>& approximate_neighbors,
    std::size_t k
)
{
    if (k == 0 || exact_neighbors.empty() || approximate_neighbors.empty()) {
        return 0.0f;
    }

    std::unordered_set<std::size_t> exact_set(
        exact_neighbors.begin(),
        exact_neighbors.begin() + static_cast<std::ptrdiff_t>(std::min(k, exact_neighbors.size()))
    );

    std::size_t common_count = 0;

    for (std::size_t i = 0;
         i < std::min(k, approximate_neighbors.size());
         ++i)
    {
        if (exact_set.contains(approximate_neighbors[i])) {
            ++common_count;
        }
    }

    return static_cast<float>(common_count) /
           static_cast<float>(std::min(k, exact_neighbors.size()));
}