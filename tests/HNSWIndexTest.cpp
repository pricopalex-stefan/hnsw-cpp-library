#include <hnsw/HnswIndex.hpp>
#include <hnsw/Node.hpp>
#include <hnsw/EuclideanDistance.hpp>

#include <algorithm>
#include <iostream>
#include <cassert>
#include <vector>

class HnswIndexTest {
    public:
        static void test_add_and_search() {
            hnsw::HnswConfig config{
                .M = 2,
                .ef_construction = 2,
                .ef_search = 2
            };
            hnsw::HnswIndex<hnsw::EuclideanDistance> index(config);

            std::size_t level = 0;
            hnsw::Node query(6, {3.5f, 4.0f}, 0);
            std::vector<hnsw::Node> nodes;
            for(int i = 0; i < 4; ++i) {
                hnsw::Node node(i, {static_cast<float>(i) + 0.5f, static_cast<float>(i) + 1.5f}, 0);
                nodes.push_back(node);
            }

            nodes[0].add_neighbor(1, level);
            nodes[0].add_neighbor(2, level);

            nodes[1].add_neighbor(0, level);
            nodes[1].add_neighbor(2, level);

            nodes[2].add_neighbor(0, level);
            nodes[2].add_neighbor(1, level);
            nodes[2].add_neighbor(3, level);

            query.add_neighbor(0, level);

            index.nodes_ = nodes;
            auto neighbors = index.search_layer(query, level);
            
            assert(neighbors.size() == config.ef_construction);
            assert(neighbors[0] == 3);
            assert(neighbors[1] == 2);
        }
};

int main()
{
    HnswIndexTest::test_add_and_search();
    return 0;
}