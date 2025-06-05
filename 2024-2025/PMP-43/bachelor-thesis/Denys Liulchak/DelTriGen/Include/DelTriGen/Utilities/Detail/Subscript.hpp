#ifndef DELTRIGEN_UTILITIES_DETAIL_SUBSCRIPT_HPP
#define DELTRIGEN_UTILITIES_DETAIL_SUBSCRIPT_HPP

#include <DelTriGen/Utilities/Concepts/Subscriptable.hpp>

namespace DelTriGen::Utilities::Detail
{

class Subscript
{
public:

    template <Concepts::Subscriptable T>
    static constexpr decltype(auto) operator()(T&& value, std::size_t index)
        noexcept(noexcept(std::forward<T>(value)[index]));
};



template<Concepts::Subscriptable T>
constexpr decltype(auto) Subscript::operator()(T&& value, std::size_t index) 
    noexcept(noexcept(std::forward<T>(value)[index]))
{
    return std::forward<T>(value)[index];
}

} // namespace DelTriGen::Utilities::Detail

#endif // DELTRIGEN_UTILITIES_DETAIL_SUBSCRIPT_HPP
