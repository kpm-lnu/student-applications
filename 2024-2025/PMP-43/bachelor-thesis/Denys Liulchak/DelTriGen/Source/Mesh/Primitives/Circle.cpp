#include <iostream>

#include <DelTriGen/Mesh/Primitives/Circle.hpp>

namespace DelTriGen::Mesh::Primitives
{

std::istream& operator>>(std::istream& is, Circle& circle)
{
    is >> circle.center_ >> circle.squaredRadius_;

    return is;
}

std::ostream& operator<<(std::ostream& os, const Circle& circle)
{
    os << circle.center_ << ' ' << circle.squaredRadius_;

    return os;
}

} // namespace DelTriGen::Mesh::Primitives
