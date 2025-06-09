#ifndef DELTRIGEN_MESH_TRIANGULATION_CONCEPTS_TRIANGULATION_HPP
#define DELTRIGEN_MESH_TRIANGULATION_CONCEPTS_TRIANGULATION_HPP

#include <DelTriGen/Mesh/Primitives/FWD.hpp>
#include <DelTriGen/Utilities/Concepts/RangeOf.hpp>

namespace DelTriGen::Mesh::Triangulation::Concepts
{

template <typename T>
concept Triangulation =
    requires(T value)
    {
        {
            [](T& mesh)
            {
                using Point    = Primitives::Point;
                using Triangle = Primitives::Triangle;

                auto& [vertices, triangles] = mesh;
               
                using VerticesT  = decltype(vertices);
                using TrianglesT = decltype(triangles);

                if constexpr
                    (Utilities::Concepts::ForwardRangeOf<VerticesT, Point> &&
                     Utilities::Concepts::ForwardRangeOf<TrianglesT, Triangle>)
                {
                    return std::true_type{};
                }
                else
                {
                    return std::false_type{};
                }
            }(value)
        } -> std::same_as<std::true_type>;
    };

} // namespace DelTriGen::Mesh::Triangulation::Concepts

#endif // DELTRIGEN_MESH_TRIANGULATION_CONCEPTS_TRIANGULATION_HPP
