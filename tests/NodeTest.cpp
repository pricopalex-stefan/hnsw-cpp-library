#include <cassert>
#include <iostream>
#include <type_traits>
#include <hnsw/Node.hpp>

int main()
{
    hnsw::Vector data{1.0f, 2.0f, 3.0f};

    hnsw::Node node(1, data, 2);

    assert(node.id() == 1);
    assert(node.level() == 2);

    data[0] = 0.0f;

    assert(node.data()[0] == 1.0f);

    static_assert(noexcept(node.id()), "node.id() should be noexcept");
    static_assert(noexcept(node.data()), "node.data() should be noexcept");
    static_assert(noexcept(node.level()), "node.level() should be noexcept");

    static_assert(std::is_same_v<
        decltype(node.data()),
        const hnsw::Vector&
        >, "node.data() should return const hnsw::Vector&");

    node.add_neighbor(2, 0);
    node.add_neighbor(3, 0);
    assert(node.neighbors(0) == std::vector<std::size_t>({2, 3}));

    std::vector<size_t> new_neighbors{3, 5};
    node.replace_neighbors(new_neighbors, 0);
    assert(node.neighbors(0) == std::vector<std::size_t>({3, 5}));
    
    return 0;
}