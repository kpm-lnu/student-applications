#ifndef DELTRIGEN_MESH_CONCEPTS_POINT_HPP
#define DELTRIGEN_MESH_CONCEPTS_POINT_HPP

#include <DelTriGen/Mesh/Primitives/Point.hpp>
#include <DelTriGen/Utilities/Concepts/ExtractorOf.hpp>

namespace DelTriGen::Mesh::Concepts
{

template <typename F>
concept PointScalarFunction =
    Utilities::Concepts::ExtractorOf<F
        , Primitives::Point::CoordinateType
        , Primitives::Point>;

} // namespace DelTriGen::Concepts

#endif // DELTRIGEN_MESH_CONCEPTS_POINT_HPP
