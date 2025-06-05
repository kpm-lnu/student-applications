#ifndef DELTRIGEN_MESH_TOOLS_CIRCUMCIRCLE_HPP
#define DELTRIGEN_MESH_TOOLS_CIRCUMCIRCLE_HPP

#include <DelTriGen/Mesh/Tools/Detail/GeometricTool.hpp>

namespace DelTriGen::Mesh::Tools
{

template <typename EntityT>
struct CircumcircleImpl;

inline constexpr Detail::GeometricTool<CircumcircleImpl> circumcircle;

} // namespace DelTriGen::Mesh::Tools

#endif // DELTRIGEN_MESH_TOOLS_CIRCUMCIRCLE_HPP
