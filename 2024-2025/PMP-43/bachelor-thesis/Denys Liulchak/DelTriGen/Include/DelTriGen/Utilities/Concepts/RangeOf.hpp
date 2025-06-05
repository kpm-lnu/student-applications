#ifndef DELTRIGEN_UTILITIES_CONCEPTS_RANGEOF_HPP
#define DELTRIGEN_UTILITIES_CONCEPTS_RANGEOF_HPP

#include <type_traits>
#include <concepts>
#include <ranges>

namespace DelTriGen::Utilities::Concepts
{

template <typename RangeT, typename Of>
concept RangeOf =
    std::same_as<std::ranges::range_value_t<RangeT>, Of>;

template <typename RangeT, typename Of>
concept InputRangeOf =
    std::ranges::input_range<RangeT> &&
    RangeOf<RangeT, Of>;

template <typename RangeT, typename Of>
concept OutputRangeOf =
    std::ranges::output_range<RangeT, Of> &&
    RangeOf<RangeT, Of>;

template <typename RangeT, typename Of>
concept ForwardRangeOf =
    std::ranges::forward_range<RangeT> &&
    RangeOf<RangeT, Of>;

template <typename RangeT, typename Of>
concept BidirectionalRangeOf =
    std::ranges::bidirectional_range<RangeT> &&
    RangeOf<RangeT, Of>;

template <typename RangeT, typename Of>
concept RandomAccessRangeOf =
    std::ranges::random_access_range<RangeT> &&
    RangeOf<RangeT, Of>;

template <typename RangeT, typename Of>
concept ContiguousRangeOf =
    std::ranges::contiguous_range<RangeT> &&
    RangeOf<RangeT, Of>;

} // namespace DelTriGen::Utilities::Concepts

#endif // DELTRIGEN_UTILITIES_CONCEPTS_RANGEOF_HPP
