#ifndef DELTRIGEN_MESH_TRIANGULATION_TOOLS_TRIANGULATOR_HPP
#define DELTRIGEN_MESH_TRIANGULATION_TOOLS_TRIANGULATOR_HPP

#include <DelTriGen/Mesh/Primitives/Triangle.hpp>
#include <DelTriGen/Utilities/NonConstructible.hpp>
#include <DelTriGen/Mesh/Triangulation/Concepts/Mesh.hpp>

namespace DelTriGen::Mesh::Triangulation::Tools
{

template <typename Derived>
class Triangulator : private Utilities::NonConstructible
{
public:

    using Point          = Primitives::Point;
    using Triangle       = Primitives::Triangle;
    using Circle         = Primitives::Circle;
    using CoordinateType = Point::CoordinateType;
    using PointIndex     = Triangle::PointIndex;
    using PointIndexes   = Triangle::PointIndexes;

    template <Utilities::Concepts::InputRangeOf<Point> RangeT>
    static constexpr auto triangulate(RangeT&& points)
        -> Concepts::Mesh auto;
};



template <typename Derived>
template <Utilities::Concepts::InputRangeOf<Primitives::Point> RangeT>
constexpr auto Triangulator<Derived>::triangulate(RangeT&& points)
    -> Concepts::Mesh auto
{
    return Derived::triangulate(std::forward<RangeT>(points));
}

} // namespace DelTriGen::Mesh::Triangulation::Tools

#endif // DELTRIGEN_MESH_TRIANGULATION_TOOLS_TRIANGULATOR_HPP
