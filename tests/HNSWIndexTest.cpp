#include <hnsw/HnswIndex.hpp>
#include <hnsw/Node.hpp>
#include <hnsw/EuclideanDistance.hpp>

#include <algorithm>
#include <iostream>
#include <cassert>
#include <vector>

class HnswIndexTest {
    public:
        static void test_search_layer() {
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
            index.global_entry_point_ = 0;
            auto neighbors = index.search_layer(query, *index.global_entry_point_, level);
            
            assert(neighbors.size() == config.ef_construction);
            assert(neighbors[0] == 3);
            assert(neighbors[1] == 2);
        }

        static void test_add() {
            hnsw::HnswConfig config{
                .M = 2,
                .ef_construction = 2,
                .ef_search = 2
            };
            hnsw::HnswIndex<hnsw::EuclideanDistance> index(config);

            assert(!index.global_entry_point_.has_value());

            index.add({0.0f, 1.0f});

            assert(index.global_entry_point_ == 0);

            index.add({1.0f, 1.0f});

            assert(index.nodes_.size() == 2);
        }
};

int main()
{
    HnswIndexTest::test_search_layer();
    HnswIndexTest::test_add();
    return 0;
}