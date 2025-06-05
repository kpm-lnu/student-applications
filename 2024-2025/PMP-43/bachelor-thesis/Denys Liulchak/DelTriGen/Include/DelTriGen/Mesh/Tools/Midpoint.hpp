#ifndef DELTRIGEN_MESH_TOOLS_MIDPOINT_HPP
#define DELTRIGEN_MESH_TOOLS_MIDPOINT_HPP

#include <DelTriGen/Mesh/Tools/Detail/GeometricTool.hpp>

namespace DelTriGen::Mesh::Tools
{

template <typename EntityT>
struct MidpointImpl;

inline constexpr Detail::GeometricTool<MidpointImpl> midpoint;

} // namespace DelTriGen::Mesh::Tools

#endif // DELTRIGEN_MESH_TOOLS_MIDPOINT_HPP
