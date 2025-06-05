#include <iostream>

#include <DelTriGen/Mesh/Primitives/Vector.hpp>

namespace DelTriGen::Mesh::Primitives
{

std::istream& operator>>(std::istream& is, Vector& vector)
{
    for (auto& coordinate : vector.coordinates_)
    {
        is >> coordinate;
    }

    return is;
}

std::ostream& operator<<(std::ostream& os, Vector vector)
{
    constexpr std::size_t lastIndex = Vector::ndims - 1;

    for (std::size_t i = 0; i < lastIndex; ++i)
    {
        os << vector[i] << ' ';
    }

    os << vector[lastIndex];

    return os;
}

} // namespace DelTriGen::Mesh::Primitives
