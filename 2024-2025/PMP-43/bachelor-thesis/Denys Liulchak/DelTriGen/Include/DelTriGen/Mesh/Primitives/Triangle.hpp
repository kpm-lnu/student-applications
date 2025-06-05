#ifndef DELTRIGEN_MESH_PRIMITIVES_TRIANGLE_HPP
#define DELTRIGEN_MESH_PRIMITIVES_TRIANGLE_HPP

#include <DelTriGen/Mesh/Primitives/Edge.hpp>
#include <DelTriGen/Utilities/Concepts/ExtractorOf.hpp>
#include <DelTriGen/Mesh/Concepts/PointScalarFunction.hpp>
#include <DelTriGen/Utilities/Subscript.hpp>
#include <DelTriGen/Mesh/Tools/Angles.hpp>
#include <DelTriGen/Mesh/Tools/Centroid.hpp>
#include <DelTriGen/Mesh/Tools/Gradient.hpp>
#include <DelTriGen/Mesh/Tools/Circumcircle.hpp>

namespace DelTriGen::Mesh::Primitives
{

class Triangle
{
public:

    static constexpr std::size_t verticesCount = 3;
    static constexpr std::size_t edgesCount    = 3;

    using PointIndex   = std::size_t;
    using PointIndexes = std::array<PointIndex, verticesCount>;
    using Edges        = std::array<Edge, edgesCount>;

    constexpr Triangle() noexcept;
    explicit constexpr Triangle(Edge edge, PointIndex index) noexcept;
    explicit constexpr Triangle(PointIndex a, PointIndex b, PointIndex c) noexcept;

    constexpr const PointIndexes& indexes() const noexcept;
    constexpr void indexes(const PointIndexes& value) noexcept;

    constexpr Edges edges() const noexcept;

    constexpr bool contains(PointIndex index) const noexcept;

    constexpr void shift(PointIndex value) noexcept;

    constexpr const PointIndex& operator[](std::size_t index) const noexcept;

    constexpr Triangle& operator=(const PointIndexes& rhs) & noexcept;

    constexpr auto operator<=>(const Triangle&) const noexcept = default;

    friend std::istream& operator>>(std::istream& is, Triangle& triangle);
    friend std::ostream& operator<<(std::ostream& os, const Triangle& triangle);

private:

    PointIndexes indexes_;
};



constexpr Triangle::Triangle() noexcept
    : indexes_{}
{}

constexpr Triangle::Triangle(Edge edge, PointIndex index) noexcept
{
    if (edge[0] > index)
    {
        indexes_ = { index, edge[0], edge[1] };
    }
    else if (edge[1] < index)
    {
        indexes_ = { edge[0], edge[1], index };
    }
    else
    {
        indexes_ = { edge[0], index, edge[1] };
    }
}

constexpr Triangle::Triangle(PointIndex a, PointIndex b, PointIndex c) noexcept
{
    indexes({ a, b, c });
}

constexpr const Triangle::PointIndexes& Triangle::indexes() const noexcept
{
    return indexes_;
}

constexpr void Triangle::indexes(const PointIndexes& value) noexcept
{
    auto [a, b, c] = value;

    if (a > b)
    {
        std::swap(a, b);
    }

    if (a > c)
    {
        std::swap(a, c);
    }

    if (b > c)
    {
        std::swap(b, c);
    }

    indexes_ = { a, b, c };
}

constexpr Triangle::Edges Triangle::edges() const noexcept
{
    return Edges
           { 
               Edge{ indexes_[0], indexes_[1] },
               Edge{ indexes_[0], indexes_[2] },
               Edge{ indexes_[1], indexes_[2] }
           };
}

constexpr bool Triangle::contains(PointIndex index) const noexcept
{
    if (index < indexes_[0] ||
        index > indexes_[2])
    {
        return false;
    }

    return indexes_[0] == index ||
           indexes_[1] == index ||
           indexes_[2] == index;
}

constexpr void Triangle::shift(PointIndex value) noexcept
{
    for (auto& index : indexes_)
    {
        index += value;
    }
}

constexpr const Triangle::PointIndex& Triangle::operator[](std::size_t index)
    const noexcept
{
    return indexes_[index];
}

constexpr Triangle& Triangle::operator=(const PointIndexes& rhs) & noexcept
{    
    indexes(rhs);

    return *this;
}

} // namespace DelTriGen::Mesh::Primitives

namespace DelTriGen::Mesh::Tools
{

template <>
struct AnglesImpl<Primitives::Triangle>
{
public:

    using Point          = Primitives::Point;
    using Vector         = Primitives::Vector;
    using Triangle       = Primitives::Triangle;
    using Subscript      = decltype(Utilities::subscript);
    using CoordinateType = Point::CoordinateType;
    using PointIndex     = Triangle::PointIndex;
    using Angles         = std::array<CoordinateType, 3>;

    template <typename SourceT, typename ExtractorT = Subscript>
    requires Utilities::Concepts::ExtractorOf<ExtractorT, Point, SourceT, PointIndex>
    constexpr Angles operator()(const Triangle& triangle
        , SourceT&& verticesSource
        , ExtractorT extractor = Utilities::subscript) const 
        noexcept(std::is_nothrow_invocable_v<ExtractorT, SourceT, PointIndex>)
    {
        Point a = std::invoke(extractor, verticesSource, triangle[0]);
        Point b = std::invoke(extractor, verticesSource, triangle[1]);
        Point c = std::invoke(extractor, verticesSource, triangle[2]);

        Vector ab = b - a;
        Vector bc = c - b;
        Vector ca = a - c;

        auto alpha = std::atan2(std::abs(ab.crossProduct(ca)), -ab.innerProduct(ca));
        auto beta  = std::atan2(std::abs(bc.crossProduct(ab)), -bc.innerProduct(ab));
        auto gamma = std::atan2(std::abs(ca.crossProduct(bc)), -ca.innerProduct(bc));

        return { alpha, beta, gamma };
    }
};

template <>
struct AreaImpl<Primitives::Triangle>
{
public:

    using Point      = Primitives::Point;
    using Vector     = Primitives::Vector;
    using Triangle   = Primitives::Triangle;
    using Subscript  = decltype(Utilities::subscript);
    using PointIndex = Triangle::PointIndex;

    template <typename SourceT, typename ExtractorT = Subscript>
    requires Utilities::Concepts::ExtractorOf<ExtractorT, Point, SourceT, PointIndex>
    constexpr auto operator()(const Triangle& triangle
        , SourceT&& verticesSource
        , ExtractorT extractor = Utilities::subscript) const 
        noexcept(std::is_nothrow_invocable_v<ExtractorT, SourceT, PointIndex>)
    {
        Point a = std::invoke(extractor, verticesSource, triangle[0]);
        Point b = std::invoke(extractor, verticesSource, triangle[1]);
        Point c = std::invoke(extractor, verticesSource, triangle[2]);

        Vector ab = b - a;
        Vector ac = c - a;

        return std::abs(ab.crossProduct(ac)) / 2;
    }
};

template <>
struct CentroidImpl<Primitives::Triangle>
{
public:

    using Point      = Primitives::Point;
    using Triangle   = Primitives::Triangle;
    using Subscript  = decltype(Utilities::subscript);
    using PointIndex = Triangle::PointIndex;

    template <typename SourceT, typename ExtractorT = Subscript>
    requires Utilities::Concepts::ExtractorOf<ExtractorT, Point, SourceT, PointIndex>
    constexpr auto operator()(const Triangle& triangle
        , SourceT&& verticesSource
        , ExtractorT extractor = Utilities::subscript) const 
        noexcept(std::is_nothrow_invocable_v<ExtractorT, SourceT, PointIndex>)
    {
        Point a = std::invoke(extractor, verticesSource, triangle[0]);
        Point b = std::invoke(extractor, verticesSource, triangle[1]);
        Point c = std::invoke(extractor, verticesSource, triangle[2]);

        return (a + b + c) / 3;
    }
};

template <>
struct GradientImpl<Primitives::Triangle>
{
public:

    using Point      = Primitives::Point;
    using Vector     = Primitives::Vector;
    using Triangle   = Primitives::Triangle;
    using Subscript  = decltype(Utilities::subscript);
    using PointIndex = Triangle::PointIndex;

    template <typename F, typename SourceT, typename ExtractorT = Subscript>
    requires Concepts::PointScalarFunction<F> &&
    Utilities::Concepts::ExtractorOf<ExtractorT, Point, SourceT, PointIndex>
    constexpr auto operator()(const Triangle& triangle
        , F&& f
        , SourceT&& verticesSource
        , ExtractorT extractor = Utilities::subscript) const 
        noexcept(std::is_nothrow_invocable_v<F, Point> &&
                 std::is_nothrow_invocable_v<ExtractorT, SourceT, PointIndex>)
    {
        Point a = std::invoke(extractor, verticesSource, triangle[0]);
        Point b = std::invoke(extractor, verticesSource, triangle[1]);
        Point c = std::invoke(extractor, verticesSource, triangle[2]);

        auto fa = std::invoke(f, a);
        auto fb = std::invoke(f, b);
        auto fc = std::invoke(f, c);

        auto deltaFbFa = fb - fa;
        auto deltaFcFa = fc - fa;

        Vector ab = b - a;
        Vector ac = c - a;

        auto denominator = ab.crossProduct(ac);

        return Vector
        {
            ( ac[1] * deltaFbFa - ab[1] * deltaFcFa) / denominator,
            (-ac[0] * deltaFbFa + ab[0] * deltaFcFa) / denominator
        };
    }
};

template <>
struct CircumcircleImpl<Primitives::Triangle>
{
public:

    using Point      = Primitives::Point;
    using Circle     = Primitives::Circle;
    using Triangle   = Primitives::Triangle;
    using Subscript  = decltype(Utilities::subscript);
    using PointIndex = Triangle::PointIndex;

    template <typename SourceT, typename ExtractorT = Subscript>
    requires Utilities::Concepts::ExtractorOf<ExtractorT, Point, SourceT, PointIndex>
    constexpr auto operator()(const Triangle& triangle
        , SourceT&& verticesSource
        , ExtractorT extractor = Utilities::subscript) const 
        noexcept(std::is_nothrow_invocable_v<ExtractorT, SourceT, PointIndex>)
    {
        Point a = std::invoke(extractor, verticesSource, triangle[0]);
        Point b = std::invoke(extractor, verticesSource, triangle[1]);
        Point c = std::invoke(extractor, verticesSource, triangle[2]);

        auto&& [x0, y0] = a.coordinates();
        auto&& [x1, y1] = b.coordinates();
        auto&& [x2, y2] = c.coordinates();

        auto x0Y0Pow2 = x0 * x0 + y0 * y0;
        auto x1y1Pow2 = x1 * x1 + y1 * y1;
        auto x2y2Pow2 = x2 * x2 + y2 * y2;

        auto deltaY1Y2 = y1 - y2;
        auto deltaY2Y0 = y2 - y0;
        auto deltaY0Y1 = y0 - y1;

        auto deltaX2X1 = x2 - x1;
        auto deltaX0X2 = x0 - x2;
        auto deltaX1X0 = x1 - x0;

        auto denominator = 2 * (x0 * deltaY1Y2 + x1 * deltaY2Y0 + x2 * deltaY0Y1);

        auto x = (x0Y0Pow2 * deltaY1Y2 + x1y1Pow2 * deltaY2Y0 + x2y2Pow2 * deltaY0Y1) /
                  denominator;

        auto y = (x0Y0Pow2 * deltaX2X1 + x1y1Pow2 * deltaX0X2 + x2y2Pow2 * deltaX1X0) /
                  denominator;

        Point center{ x, y };
        auto squaredRadius = center.squaredDistance(a);

        return Circle{ center, squaredRadius };
    }
};

} // namespace DelTriGen::Mesh::Tools

template <>
struct std::hash<DelTriGen::Mesh::Primitives::Triangle>
{
public:

    using Triangle     = DelTriGen::Mesh::Primitives::Triangle;
    using PointIndex   = Triangle::PointIndex;
    using PointIndexes = Triangle::PointIndexes;

    std::size_t operator()(const Triangle& triangle) const noexcept
    {
        const PointIndexes& indexes = triangle.indexes();

        return boost::hash_range(indexes.cbegin(), indexes.cend());
    }
};

#endif // DELTRIGEN_MESH_PRIMITIVES_TRIANGLE_HPP
