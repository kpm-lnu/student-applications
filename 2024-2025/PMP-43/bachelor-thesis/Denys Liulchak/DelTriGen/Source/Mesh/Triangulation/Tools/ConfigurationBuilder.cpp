#include <limits>

#include <DelTriGen/Mesh/Triangulation/Tools/ConfigurationBuilder.hpp>

namespace DelTriGen::Mesh::Triangulation::Tools
{

ConfigurationBuilder& ConfigurationBuilder::removeSuperstructure(bool value) noexcept
{
    config_.removeSuperstructure = value;

    return *this;
}

ConfigurationBuilder& ConfigurationBuilder::strategy(RefinementStrategy value) noexcept
{
    config_.strategy = value;

    return *this;
}

ConfigurationBuilder& ConfigurationBuilder::maxArea(CoordinateType value) noexcept
{
    config_.maxArea = value > 0 ? value
                                : std::numeric_limits<CoordinateType>::infinity();

    return *this;
}

ConfigurationBuilder& ConfigurationBuilder::maxEdgeLength(CoordinateType value) noexcept
{
    config_.maxEdgeLength = value > 0 ? value
                                      : std::numeric_limits<CoordinateType>::infinity();

    return *this;
}

ConfigurationBuilder& ConfigurationBuilder::areaThreshold(CoordinateType value) noexcept
{
    config_.areaThreshold = value > 0 ? value
                                      : std::numeric_limits<CoordinateType>::infinity();

    return *this;
}

ConfigurationBuilder& ConfigurationBuilder::gradThreshold(CoordinateType value) noexcept
{
    config_.gradThreshold = value > 0 ? value
                                      : std::numeric_limits<CoordinateType>::infinity();

    return *this;
}

const Configuration ConfigurationBuilder::build() noexcept
{
    return std::move(config_);
}

} // namespace DelTriGen::Mesh::Triangulation::Tools
