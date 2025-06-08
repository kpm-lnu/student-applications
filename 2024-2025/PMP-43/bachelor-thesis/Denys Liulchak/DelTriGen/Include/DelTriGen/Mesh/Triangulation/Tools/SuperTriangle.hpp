#ifndef DELTRIGEN_MESH_TRIANGULATION_TOOLS_SUPERTRIANGLE_HPP
#define DELTRIGEN_MESH_TRIANGULATION_TOOLS_SUPERTRIANGLE_HPP

#include <limits>

#include <DelTriGen/Mesh/Tools/MinMaxCoordinates.hpp>
#include <DelTriGen/Mesh/Triangulation/Tools/SuperstructureGenerator.hpp>

namespace DelTriGen::Mesh::Triangulation::Tools
{

class SuperTriangle : public SuperstructureGenerator<SuperTriangle>
{
public:

    static constexpr std::size_t verticesCount  = 3;
    static constexpr std::size_t trianglesCount = 1;

    using Points         = std::array<Point, verticesCount>;
    using Triangles      = std::array<Triangle, trianglesCount>;
    using Superstructure = std::pair<Points, Triangles>;

    template <Utilities::Concepts::ForwardRangeOf<Point> RangeT>
    static constexpr Superstructure generate(RangeT&& points);

private:

    static constexpr Superstructure infTriangle() noexcept;
};



template <Utilities::Concepts::ForwardRangeOf<SuperTriangle::Point> RangeT>
constexpr SuperTriangle::Superstructure SuperTriangle::generate(RangeT&& points)
{
    if (std::ranges::empty(points))
    {
        return infTriangle();
    }

    auto&& [minX, minY, maxX, maxY] = Mesh::Tools::minMaxCoordinates(points);

    CoordinateType midX  = std::midpoint(minX, maxX);
    CoordinateType dx    = maxX - minX;
    CoordinateType dy    = maxY - minY;
    CoordinateType shift = std::ranges::max(dx, dy);

    if (!std::isfinite(shift))
    {
        return infTriangle();
    }
    else if (shift < std::numeric_limits<CoordinateType>::epsilon())
    {
        shift = 1;
    }

    minY -= 0.2 * shift;

    Points vertices =
    {
        Point{ minX - shift, minY },
        Point{ maxX + shift, minY },
        Point{ midX, maxY + shift }
    };

    Triangles triangles =
    {
        Triangle{ 0, 1, 2 }
    };

    return Superstructure{ std::move(vertices), std::move(triangles) };
}

constexpr SuperTriangle::Superstructure SuperTriangle::infTriangle() noexcept
{
    CoordinateType min = -std::numeric_limits<CoordinateType>::infinity();
    CoordinateType max =  std::numeric_limits<CoordinateType>::infinity();
    CoordinateType mid =  0;

    Points vertices =
    {
        Point{ min, min },
        Point{ max, min },
        Point{ mid, max }
    };

    Triangles triangles =
    {
        Triangle{ 0, 1, 2 }
    };

    return Superstructure{ std::move(vertices), std::move(triangles) };
}

} // namespace DelTriGen::Mesh::Triangulation::Tools

#endif // DELTRIGEN_MESH_TRIANGULATION_TOOLS_SUPERTRIANGLE_HPP
