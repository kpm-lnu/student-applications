#include <iostream>

#include <DelTriGen/Mesh/Primitives/Edge.hpp>

namespace DelTriGen::Mesh::Primitives
{

std::istream& operator>>(std::istream& is, Edge& edge)
{
    for (auto& index : edge.indexes_)
    {
        is >> index;
    }

    edge.indexes(edge.indexes_);

    return is;
}

std::ostream& operator<<(std::ostream& os, Edge edge)
{
    constexpr std::size_t lastIndex = Edge::verticesCount - 1;

    for (std::size_t i = 0; i < lastIndex; ++i)
    {
        os << edge[i] << ' ';
    }

    os << edge[lastIndex];

    return os;
}

} // namespace DelTriGen::Mesh::Primitives
