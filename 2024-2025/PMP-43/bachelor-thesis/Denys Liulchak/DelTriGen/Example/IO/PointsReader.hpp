#ifndef EXAMPLE_IO_POINTSREADER_HPP
#define EXAMPLE_IO_POINTSREADER_HPP

#include <ranges>

#include <DelTriGen/Mesh/Primitives/Point.hpp>
#include <DelTriGen/Utilities/NonConstructible.hpp>

namespace Example::IO
{

class PointsReader : private DelTriGen::Utilities::NonConstructible
{
public:

    using Point             = DelTriGen::Mesh::Primitives::Point;
    using PointsIstreamView = std::ranges::istream_view<Point>;

    static constexpr PointsIstreamView makeView(std::istream& is) noexcept;
};



constexpr PointsReader::PointsIstreamView PointsReader::makeView(std::istream& is)
    noexcept
{
    return PointsIstreamView(is);
}

} // namespace Example::IO

#endif // EXAMPLE_IO_POINTSREADER_HPP
