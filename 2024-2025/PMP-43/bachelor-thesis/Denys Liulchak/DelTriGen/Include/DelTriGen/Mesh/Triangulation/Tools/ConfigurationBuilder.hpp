#ifndef DELTRIGEN_MESH_TRIANGULATION_CONFIGURATIONBUILDER_HPP
#define DELTRIGEN_MESH_TRIANGULATION_CONFIGURATIONBUILDER_HPP

#include <DelTriGen/Mesh/Triangulation/Configuration.hpp>
#include <DelTriGen/Mesh/Concepts/PointScalarFunction.hpp>

namespace DelTriGen::Mesh::Triangulation::Tools
{

class ConfigurationBuilder
{
public:

    using CoordinateType = Configuration::CoordinateType;

    ConfigurationBuilder() noexcept = default;

    ConfigurationBuilder& removeSuperstructure(bool value) noexcept;
    ConfigurationBuilder& strategy(RefinementStrategy value) noexcept;
    ConfigurationBuilder& maxArea(CoordinateType value) noexcept;
    ConfigurationBuilder& maxEdgeLength(CoordinateType value) noexcept;
    ConfigurationBuilder& areaThreshold(CoordinateType value) noexcept;
    ConfigurationBuilder& gradThreshold(CoordinateType value) noexcept;

    template <Mesh::Concepts::PointScalarFunction F>
    ConfigurationBuilder& f(F&& value) noexcept(std::is_rvalue_reference_v<F&&>);

    const Configuration build() noexcept;

private:

    Configuration config_;
};



template<Mesh::Concepts::PointScalarFunction F>
ConfigurationBuilder& ConfigurationBuilder::f(F&& value)
    noexcept(std::is_rvalue_reference_v<F&&>)
{
    config_.f = std::forward<F>(value);

    return *this;
}

} // namespace DelTriGen::Mesh::Triangulation::Tools

#endif // DELTRIGEN_MESH_TRIANGULATION_CONFIGURATIONBUILDER_HPP
