#ifndef DELTRIGEN_MESH_TRIANGULATION_CONFIGURATION_HPP
#define DELTRIGEN_MESH_TRIANGULATION_CONFIGURATION_HPP

#include <DelTriGen/Mesh/Primitives/Point.hpp>
#include <DelTriGen/Mesh/Triangulation/Tools/FWD.hpp>

namespace DelTriGen::Mesh::Triangulation
{;

enum class RefinementStrategy : std::uint8_t
{
    None       = 0,
    Area       = 1,
    EdgeLength = 2,
    Boundary   = 3,
    Gradient   = 4
};

struct Configuration
{
public:

    using Point               = Primitives::Point;
    using CoordinateType      = Point::CoordinateType;
    using PointScalarFunction = std::function<CoordinateType(Point)>;

    bool removeSuperstructure;
    RefinementStrategy strategy;
    CoordinateType maxArea;
    CoordinateType maxEdgeLength;
    CoordinateType areaThreshold;
    CoordinateType gradThreshold;
    PointScalarFunction f;

    friend class Tools::ConfigurationBuilder;

private:

    Configuration() noexcept = default;
};

} // namespace DelTriGen::Mesh::Triangulation

#endif // DELTRIGEN_MESH_TRIANGULATION_CONFIGURATION_HPP
