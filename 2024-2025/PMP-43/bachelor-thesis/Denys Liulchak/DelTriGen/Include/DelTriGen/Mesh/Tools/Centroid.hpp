#ifndef DELTRIGEN_MESH_TOOLS_CENTROID_HPP
#define DELTRIGEN_MESH_TOOLS_CENTROID_HPP

#include <DelTriGen/Mesh/Tools/Detail/GeometricTool.hpp>

namespace DelTriGen::Mesh::Tools
{

template <typename EntityT>
struct CentroidImpl;

inline constexpr Detail::GeometricTool<CentroidImpl> centroid;

} // namespace DelTriGen::Mesh::Tools

#endif // DELTRIGEN_MESH_TOOLS_CENTROID_HPP
