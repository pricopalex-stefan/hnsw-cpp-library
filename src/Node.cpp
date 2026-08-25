#include <hnsw/Vector.hpp>
#include <hnsw/Node.hpp>
#include <utility>

namespace hnsw {

    Node::Node(int id, Vector data, int level)
        : id_(id),
        data_(std::move(data)),
        level_(level) 
    {

    }

    int Node::id() const noexcept
    {
        return id_;
    }

    const Vector& Node::data() const noexcept
    {
        return data_;
    }

    int Node::level() const noexcept
    {
        return level_;
    }

}