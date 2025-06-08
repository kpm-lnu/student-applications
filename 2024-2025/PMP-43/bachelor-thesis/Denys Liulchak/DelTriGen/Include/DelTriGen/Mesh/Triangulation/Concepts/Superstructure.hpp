#ifndef DELTRIGEN_MESH_TRIANGULATION_CONCEPTS_SUPERSTRUCTURE_HPP
#define DELTRIGEN_MESH_TRIANGULATION_CONCEPTS_SUPERSTRUCTURE_HPP

#include <DelTriGen/Mesh/Triangulation/Tools/FWD.hpp>
#include <DelTriGen/Mesh/Triangulation/Concepts/Triangulation.hpp>

namespace DelTriGen::Mesh::Triangulation::Concepts
{

template <typename RangeT, typename T>
concept Superstructure =
    Utilities::Concepts::ForwardRangeOf<RangeT, Primitives::Point> &&
    std::derived_from<T, Tools::SuperstructureGenerator<T>>        &&
    requires(RangeT&& range)
    {
        { T::generate(range) } -> Triangulation;
    };

} // namespace DelTriGen::Mesh::Triangulation::Concepts

#endif // DELTRIGEN_MESH_TRIANGULATION_CONCEPTS_SUPERSTRUCTURE_HPP
