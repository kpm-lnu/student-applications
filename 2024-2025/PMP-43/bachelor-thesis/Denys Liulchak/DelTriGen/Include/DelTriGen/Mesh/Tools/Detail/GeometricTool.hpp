#ifndef DELTRIGEN_MESH_TOOLS_DETAIL_GEOMETRICTOOL_HPP
#define DELTRIGEN_MESH_TOOLS_DETAIL_GEOMETRICTOOL_HPP

#include <utility>
#include <type_traits>
#include <concepts>

namespace DelTriGen::Mesh::Tools::Detail
{

template <typename T>
using Entity = std::remove_cvref_t<T>;

template <template <typename> typename Implementer, typename T, typename... Args>
concept ImplExists =
    std::default_initializable<Implementer<Entity<T>>> &&
    std::invocable<Implementer<Entity<T>>, T, Args...>;

template <template <typename> typename Implementer>
class GeometricTool
{
public:

    template <typename T, typename... Args>
    requires ImplExists<Implementer, T, Args...>
    static constexpr decltype(auto) operator()(T&& entity, Args&&... args)
        noexcept(noexcept(
            Implementer<Entity<T>>()
            (std::forward<T>(entity), std::forward<Args>(args)...)
        ));
};



template <template <typename> typename Implementer>
template<typename T, typename... Args>
requires ImplExists<Implementer, T, Args...>
constexpr decltype(auto) GeometricTool<Implementer>::operator()(T&& entity, Args&&... args)
    noexcept(noexcept(
        Implementer<Entity<T>>()
        (std::forward<T>(entity), std::forward<Args>(args)...)
    ))
{
    return Implementer<Entity<T>>()
        (std::forward<T>(entity), std::forward<Args>(args)...);
}

} // namespace DelTriGen::Mesh::Tools::Detail

#endif // DELTRIGEN_MESH_TOOLS_DETAIL_GEOMETRICTOOL_HPP
