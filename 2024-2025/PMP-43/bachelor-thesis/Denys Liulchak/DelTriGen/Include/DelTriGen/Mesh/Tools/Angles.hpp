#ifndef DELTRIGEN_MESH_TOOLS_ANGLES_HPP
#define DELTRIGEN_MESH_TOOLS_ANGLES_HPP

#include <DelTriGen/Mesh/Tools/Detail/GeometricTool.hpp>

namespace DelTriGen::Mesh::Tools
{

template <typename EntityT>
struct AnglesImpl;

inline constexpr Detail::GeometricTool<AnglesImpl> angles;

} // namespace DelTriGen::Mesh::Tools

#endif // DELTRIGEN_MESH_TOOLS_ANGLES_HPP
