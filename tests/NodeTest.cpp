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
    
    return 0;
}