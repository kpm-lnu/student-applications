#ifndef DELTRIGEN_MESH_TOOLS_SQUAREDLENGTH_HPP
#define DELTRIGEN_MESH_TOOLS_SQUAREDLENGTH_HPP

#include <DelTriGen/Mesh/Tools/Detail/GeometricTool.hpp>

namespace DelTriGen::Mesh::Tools
{

template <typename EntityT>
struct SquaredLengthImpl;

inline constexpr Detail::GeometricTool<SquaredLengthImpl> squaredLength;

} // namespace DelTriGen::Mesh::Tools

#endif // DELTRIGEN_MESH_TOOLS_SQUAREDLENGTH_HPP
