#ifndef EXAMPLE_IO_RANGEWRITER_HPP
#define EXAMPLE_IO_RANGEWRITER_HPP

#include <iosfwd>
#include <ranges>

#include <DelTriGen/Utilities/NonConstructible.hpp>

namespace Example::IO
{

template <typename T>
concept OStreamWritableRange =
    std::ranges::input_range<T> &&
    requires(std::ostream& os, std::ranges::range_reference_t<T> value)
    {
        os << value;
    };


class RangeWriter : private DelTriGen::Utilities::NonConstructible
{
public:

    template <OStreamWritableRange RangeT>
    static void write(std::ostream& os, RangeT&& range);
};



template <OStreamWritableRange RangeT>
void RangeWriter::write(std::ostream& os, RangeT&& range)
{
    auto first = std::ranges::begin(range);
    auto last  = std::ranges::end(range);

    if (first == last)
    {
        return;
    }

    os << *first;

    while (++first != last)
    {
        os << '\n' << *first;
    }
}

} // namespace Example::IO

#endif // EXAMPLE_IO_RANGEWRITER_HPP
