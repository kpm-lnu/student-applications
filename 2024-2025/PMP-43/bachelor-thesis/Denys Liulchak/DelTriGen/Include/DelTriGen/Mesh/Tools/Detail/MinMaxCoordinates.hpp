#ifndef DELTRIGEN_MESH_TOOLS_DETAIL_MINMAXCOORDINATES_HPP
#define DELTRIGEN_MESH_TOOLS_DETAIL_MINMAXCOORDINATES_HPP

#include <DelTriGen/Mesh/Primitives/Point.hpp>
#include <DelTriGen/Utilities/Concepts/RangeOf.hpp>

namespace DelTriGen::Mesh::Tools::Detail
{

struct MinMaxCoordinatesResult
{
public:

    using Point          = Primitives::Point;
    using CoordinateType = Point::CoordinateType;

    CoordinateType minX;
    CoordinateType minY;
    CoordinateType maxX;
    CoordinateType maxY;
};

class MinMaxCoordinates
{
public:

    using Point = Primitives::Point;

    template <Utilities::Concepts::ForwardRangeOf<Point> RangeT>
    static constexpr MinMaxCoordinatesResult operator()(RangeT&& points);
};



template <Utilities::Concepts::ForwardRangeOf<MinMaxCoordinates::Point> RangeT>
constexpr MinMaxCoordinatesResult MinMaxCoordinates::operator()(RangeT&& points)
{
    auto byX = [](Point point) noexcept { return point[0]; };
    auto byY = [](Point point) noexcept { return point[1]; };

    auto&& boundsX = std::ranges::minmax(points, {}, byX);
    auto&& boundsY = std::ranges::minmax(points, {}, byY);

    return
    {
        boundsX.min[0],
        boundsY.min[1],
        boundsX.max[0],
        boundsY.max[1]
    };
}

} // namespace DelTriGen::Mesh::Tools::Detail

#endif // DELTRIGEN_MESH_TOOLS_DETAIL_MINMAXCOORDINATES_HPP
