#ifndef DELTRIGEN_MESH_TOOLS_AREA_HPP
#define DELTRIGEN_MESH_TOOLS_AREA_HPP

#include <DelTriGen/Mesh/Tools/Detail/GeometricTool.hpp>

namespace DelTriGen::Mesh::Tools
{

template <typename EntityT>
struct AreaImpl;

inline constexpr Detail::GeometricTool<AreaImpl> area;

} // namespace DelTriGen::Mesh::Tools

#endif // DELTRIGEN_MESH_TOOLS_AREA_HPP
