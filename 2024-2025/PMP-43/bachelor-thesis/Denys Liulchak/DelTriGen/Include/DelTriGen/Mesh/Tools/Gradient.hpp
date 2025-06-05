#ifndef DELTRIGEN_MESH_TOOLS_GRADIENT_HPP
#define DELTRIGEN_MESH_TOOLS_GRADIENT_HPP

#include <DelTriGen/Mesh/Tools/Detail/GeometricTool.hpp>

namespace DelTriGen::Mesh::Tools
{

template <typename EntityT>
struct GradientImpl;

inline constexpr Detail::GeometricTool<GradientImpl> gradient;

} // namespace DelTriGen::Mesh::Tools

#endif // DELTRIGEN_MESH_TOOLS_GRADIENT_HPP
