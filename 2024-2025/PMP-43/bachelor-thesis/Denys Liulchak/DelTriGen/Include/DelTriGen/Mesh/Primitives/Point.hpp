#ifndef DELTRIGEN_MESH_PRIMITIVES_POINT_HPP
#define DELTRIGEN_MESH_PRIMITIVES_POINT_HPP

#include <DelTriGen/Mesh/Primitives/Vector.hpp>
#include <DelTriGen/Mesh/Tools/Midpoint.hpp>

namespace DelTriGen::Mesh::Primitives
{

using Point = Vector;

} // namespace DelTriGen::Mesh::Primitives

namespace DelTriGen::Mesh::Tools
{

template <>
struct MidpointImpl<Primitives::Point>
{
public:

    using Point = Primitives::Point;

    constexpr auto operator()(Point a, Point b) const noexcept
    {
        auto&& [x0, y0] = a.coordinates();
        auto&& [x1, y1] = b.coordinates();

        return Point{ std::midpoint(x0, x1), std::midpoint(y0, y1) };
    }
};

} // namespace DelTriGen::Mesh::Tools

#endif // DELTRIGEN_MESH_PRIMITIVES_POINT_HPP
