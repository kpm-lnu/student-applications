#ifndef DELTRIGEN_MESH_TRIANGULATION_TOOLS_SUPERRECTANGLE_HPP
#define DELTRIGEN_MESH_TRIANGULATION_TOOLS_SUPERRECTANGLE_HPP

#include <limits>

#include <DelTriGen/Mesh/Tools/MinMaxCoordinates.hpp>
#include <DelTriGen/Mesh/Triangulation/Tools/SuperstructureGenerator.hpp>

namespace DelTriGen::Mesh::Triangulation::Tools
{

class SuperRectangle : public SuperstructureGenerator<SuperRectangle>
{
public:

    static constexpr std::size_t verticesCount  = 4;
    static constexpr std::size_t trianglesCount = 2;

    using Points         = std::array<Point, verticesCount>;
    using Triangles      = std::array<Triangle, trianglesCount>;
    using Superstructure = std::pair<Points, Triangles>;

    template <Utilities::Concepts::ForwardRangeOf<Point> RangeT>
    static constexpr Superstructure generate(RangeT&& points);

private:

    static constexpr Superstructure infRectangle() noexcept;
};



template <Utilities::Concepts::ForwardRangeOf<SuperRectangle::Point> RangeT>
constexpr SuperRectangle::Superstructure SuperRectangle::generate(RangeT&& points)
{
    if (std::ranges::empty(points))
    {
        return infRectangle();
    }

    auto&& [minX, minY, maxX, maxY] = Mesh::Tools::minMaxCoordinates(points);
    
    CoordinateType dx    = maxX - minX;
    CoordinateType dy    = maxY - minY;
    CoordinateType shift = 0.1 * std::ranges::max(dx, dy);

    if (!std::isfinite(shift))
    {
        return infRectangle();
    }
    else if (shift < std::numeric_limits<CoordinateType>::epsilon())
    {
        shift = 1;
    }

    minX -= shift;
    minY -= shift;

    maxX += shift;
    maxY += shift;

    Points vertices =
    {
        Point{ minX, minY },
        Point{ maxX, minY },
        Point{ maxX, maxY },
        Point{ minX, maxY }
    };

    Triangles triangles =
    {
        Triangle{ 0, 1, 3 },
        Triangle{ 1, 2, 3 }
    };

    return Superstructure{ std::move(vertices), std::move(triangles) };
}

constexpr SuperRectangle::Superstructure SuperRectangle::infRectangle() noexcept
{
    CoordinateType min = -std::numeric_limits<CoordinateType>::infinity();
    CoordinateType max =  std::numeric_limits<CoordinateType>::infinity();

    Points vertices =
    {
        Point{ min, min },
        Point{ max, min },
        Point{ max, max },
        Point{ min, max }
    };

    Triangles triangles =
    {
        Triangle{ 0, 1, 3 },
        Triangle{ 1, 2, 3 }
    };

    return Superstructure{ std::move(vertices), std::move(triangles) };
}

} // namespace DelTriGen::Mesh::Triangulation::Tools

#endif // DELTRIGEN_MESH_TRIANGULATION_TOOLS_SUPERRECTANGLE_HPP
