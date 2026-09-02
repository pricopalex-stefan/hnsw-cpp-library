#include <hnsw/Node.hpp>
#include <utility>
#include <queue>

namespace hnsw {

    Node::Node(std::size_t id, Vector data, std::size_t level)
        : id_(id),
        data_(std::move(data)),
        level_(level),
        neighbors_(level + 1)
    {

    }

    std::size_t Node::id() const noexcept
    {
        return id_;
    }

    const Vector& Node::data() const noexcept
    {
        return data_;
    }

    std::size_t Node::level() const noexcept
    {
        return level_;
    }

    const std::vector<std::size_t>& Node::neighbors(std::size_t level) const noexcept
    {
        return neighbors_[level];
    }  

    void Node::add_neighbor(std::size_t neighbor_id, std::size_t level) noexcept
    {
        neighbors_[level].push_back(neighbor_id);
    }

    void Node::replace_neighbors(std::vector<std::size_t> neighbors, std::size_t level) 
    {
        neighbors_[level] = neighbors;
    }
}