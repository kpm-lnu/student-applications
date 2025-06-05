#ifndef DELTRIGEN_MESH_TOOLS_LENGTH_HPP
#define DELTRIGEN_MESH_TOOLS_LENGTH_HPP

#include <DelTriGen/Mesh/Tools/Detail/GeometricTool.hpp>

namespace DelTriGen::Mesh::Tools
{

template <typename EntityT>
struct LengthImpl;

inline constexpr Detail::GeometricTool<LengthImpl> length;

} // namespace DelTriGen::Mesh::Tools

#endif // DELTRIGEN_MESH_TOOLS_LENGTH_HPP
