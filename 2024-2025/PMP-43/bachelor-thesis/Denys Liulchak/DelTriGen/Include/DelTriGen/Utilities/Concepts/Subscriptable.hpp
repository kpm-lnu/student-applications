#ifndef DELTRIGEN_UTILITIES_CONCEPTS_SUBSCRIPTABLE_HPP
#define DELTRIGEN_UTILITIES_CONCEPTS_SUBSCRIPTABLE_HPP

#include <cstddef>
#include <utility>

namespace DelTriGen::Utilities::Concepts
{

template <typename T>
concept Subscriptable = 
    requires(T&& value, std::size_t index)
    {
        std::forward<T>(value)[index];
    };

} // namespace DelTriGen::Utilities::Concepts

#endif // DELTRIGEN_UTILITIES_CONCEPTS_SUBSCRIPTABLE_HPP
