#include <iostream>

#include <DelTriGen/Mesh/Primitives/Triangle.hpp>

namespace DelTriGen::Mesh::Primitives
{

std::istream& operator>>(std::istream& is, Triangle& triangle)
{
    for (auto& index : triangle.indexes_)
    {
        is >> index;
    }

    triangle.indexes(triangle.indexes_);

    return is;
}

std::ostream& operator<<(std::ostream& os, const Triangle& triangle)
{
    constexpr std::size_t lastIndex = Triangle::verticesCount - 1;

    for (std::size_t i = 0; i < lastIndex; ++i)
    {
        os << triangle[i] << ' ';
    }

    os << triangle[lastIndex];

    return os;
}

} // namespace DelTriGen::Mesh::Primitives
