#ifndef DELTRIGEN_MESH_TRIANGULATION_TOOLS_SUPERSTRUCTUREGENERATOR_HPP
#define DELTRIGEN_MESH_TRIANGULATION_TOOLS_SUPERSTRUCTUREGENERATOR_HPP

#include <DelTriGen/Mesh/Primitives/Triangle.hpp>
#include <DelTriGen/Utilities/NonConstructible.hpp>
#include <DelTriGen/Mesh/Triangulation/Concepts/Superstructure.hpp>

namespace DelTriGen::Mesh::Triangulation::Tools
{

template <typename Derived>
class SuperstructureGenerator : private Utilities::NonConstructible
{
public:

    using Point          = Primitives::Point;
    using CoordinateType = Point::CoordinateType;
    using Triangle       = Primitives::Triangle;
    using PointIndex     = Triangle::PointIndex;

    template <Concepts::Superstructure<Derived> RangeT>
    static constexpr auto generate(RangeT&& points);
};



template <typename Derived>
template <Concepts::Superstructure<Derived> RangeT>
constexpr auto SuperstructureGenerator<Derived>::generate(RangeT&& points)
{
    return Derived::generate(points);
}

} // namespace DelTriGen::Mesh::Triangulation::Tools

#endif // DELTRIGEN_MESH_TRIANGULATION_TOOLS_SUPERSTRUCTUREGENERATOR_HPP
