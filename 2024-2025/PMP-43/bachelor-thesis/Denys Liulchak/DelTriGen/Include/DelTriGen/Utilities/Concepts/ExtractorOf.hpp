#ifndef DELTRIGEN_UTILITIES_CONCEPTS_EXTRACTOROF_HPP
#define DELTRIGEN_UTILITIES_CONCEPTS_EXTRACTOROF_HPP

#include <type_traits>
#include <concepts>

namespace DelTriGen::Utilities::Concepts
{

template <typename T, typename Of, typename... From>
concept ExtractorOf =
    std::invocable<T, From...> &&
    std::convertible_to<std::invoke_result_t<T, From...>, Of>;

} // namespace DelTriGen::Utilities::Concepts

#endif // DELTRIGEN_UTILITIES_CONCEPTS_EXTRACTOROF_HPP
