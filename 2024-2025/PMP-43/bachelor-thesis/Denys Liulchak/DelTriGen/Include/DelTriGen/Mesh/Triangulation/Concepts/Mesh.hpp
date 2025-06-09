#ifndef DELTRIGEN_MESH_TRIANGULATION_CONCEPTS_MESH_HPP
#define DELTRIGEN_MESH_TRIANGULATION_CONCEPTS_MESH_HPP

#include <DelTriGen/Mesh/Triangulation/Concepts/Triangulation.hpp>

namespace DelTriGen::Mesh::Triangulation::Concepts
{

template <typename T>
concept Mesh =
    Triangulation<T> ||
    requires(T value)
    {
        {
            [](T& mesh)
            {
                using Point    = Primitives::Point;
                using Triangle = Primitives::Triangle;

                auto& [vertices, construction] = mesh;

                using VerticesT     = decltype(vertices);
                using ConstructionT = decltype(construction);
                using RangeValueT   = std::ranges::range_value_t<ConstructionT>;
                using FirstT        = decltype(std::get<0>(std::declval<RangeValueT>()));

                if constexpr
                    (Utilities::Concepts::ForwardRangeOf<VerticesT, Point> &&
                     std::ranges::forward_range<ConstructionT>             &&
                     std::convertible_to<FirstT, Triangle>)
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

#endif // DELTRIGEN_MESH_TRIANGULATION_CONCEPTS_MESH_HPP
