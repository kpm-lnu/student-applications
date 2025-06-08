#ifndef DELTRIGEN_MESH_TRIANGULATION_TOOLS_DELAUNAYCONDITION_HPP
#define DELTRIGEN_MESH_TRIANGULATION_TOOLS_DELAUNAYCONDITION_HPP

#include <DelTriGen/Mesh/Primitives/Circle.hpp>
#include <DelTriGen/Utilities/NonConstructible.hpp>

namespace DelTriGen::Mesh::Triangulation::Tools
{

class DelaunayCondition : private Utilities::NonConstructible
{
public:

    using Point          = Primitives::Point;
    using Circle         = Primitives::Circle;
    using CoordinateType = Point::CoordinateType;

    static constexpr bool isSatisfied(const Circle& circle, Point point) noexcept;
    static constexpr bool isAlmostSatisfied(const Circle& circle, Point point
        , CoordinateType absTol = 1E-13
        , CoordinateType relTol = 1E-13) noexcept;
};



constexpr bool DelaunayCondition::isSatisfied(const Circle& circle, Point point) noexcept
{
    return !circle.isInside(point);
}

constexpr bool DelaunayCondition::isAlmostSatisfied(const Circle& circle, Point point
    , CoordinateType absTol
    , CoordinateType relTol) noexcept
{
    return circle.isOutside(point) ||
           circle.isAlmostOnBoundary(point, absTol, relTol);
}

} // namespace DelTriGen::Mesh::Triangulation::Tools

#endif // DELTRIGEN_MESH_TRIANGULATION_TOOLS_DELAUNAYCONDITION_HPP
