#ifndef DELTRIGEN_MESH_PRIMITIVES_EDGE_HPP
#define DELTRIGEN_MESH_PRIMITIVES_EDGE_HPP

#include <DelTriGen/Mesh/Primitives/Circle.hpp>
#include <DelTriGen/Utilities/Concepts/ExtractorOf.hpp>
#include <DelTriGen/Utilities/Subscript.hpp>
#include <DelTriGen/Mesh/Tools/Circumcircle.hpp>

namespace DelTriGen::Mesh::Primitives
{

class Edge
{
public:

    static constexpr std::size_t verticesCount = 2;

    using PointIndex   = std::size_t;
    using PointIndexes = std::array<PointIndex, verticesCount>;

    constexpr Edge() noexcept;
    explicit constexpr Edge(PointIndex a, PointIndex b) noexcept;

    constexpr PointIndexes indexes() const noexcept;
    constexpr void indexes(PointIndexes value) noexcept;

    constexpr bool contains(PointIndex index) const noexcept;

    constexpr void shift(PointIndex value) noexcept;

    constexpr const PointIndex& operator[](std::size_t index) const noexcept;

    constexpr Edge& operator=(PointIndexes rhs) & noexcept;

    constexpr auto operator<=>(const Edge&) const noexcept = default;

    friend std::istream& operator>>(std::istream& is, Edge& edge);
    friend std::ostream& operator<<(std::ostream& os, Edge edge);

private:

    PointIndexes indexes_;
};



constexpr Edge::Edge() noexcept
    : indexes_{}
{}

constexpr Edge::Edge(PointIndex a, PointIndex b) noexcept
{
    indexes({ a, b });
}

constexpr Edge::PointIndexes Edge::indexes() const noexcept
{
    return indexes_;
}

constexpr void Edge::indexes(PointIndexes value) noexcept
{
    auto& [a, b] = value;

    if (a < b)
    {
        indexes_ = { a, b };
    }
    else
    {
        indexes_ = { b, a };
    }
}

constexpr bool Edge::contains(PointIndex index) const noexcept
{
    return indexes_[0] == index ||
           indexes_[1] == index;
}

constexpr void Edge::shift(PointIndex value) noexcept
{
    for (auto& index : indexes_)
    {
        index += value;
    }
}

constexpr const Edge::PointIndex& Edge::operator[](std::size_t index) const noexcept
{
    return indexes_[index];
}

constexpr Edge& Edge::operator=(PointIndexes rhs) & noexcept
{
    indexes(rhs);

    return *this;
}

} // namespace DelTriGen::Mesh::Primitives

namespace DelTriGen::Mesh::Tools
{

template <>
struct LengthImpl<Primitives::Edge>
{
public:

    using Point      = Primitives::Point;
    using Edge       = Primitives::Edge;
    using Subscript  = decltype(Utilities::subscript);
    using PointIndex = Edge::PointIndex;

    template <typename SourceT, typename ExtractorT = Subscript>
    requires Utilities::Concepts::ExtractorOf<ExtractorT, Point, SourceT, PointIndex>
    constexpr auto operator()(Edge edge
        , SourceT&& verticesSource
        , ExtractorT extractor = Utilities::subscript) const 
        noexcept(std::is_nothrow_invocable_v<ExtractorT, SourceT, PointIndex>)
    {
        Point a = std::invoke(extractor, verticesSource, edge[0]);
        Point b = std::invoke(extractor, verticesSource, edge[1]);

        return a.distance(b);
    }
};

template <>
struct SquaredLengthImpl<Primitives::Edge>
{
public:

    using Point      = Primitives::Point;
    using Edge       = Primitives::Edge;
    using Subscript  = decltype(Utilities::subscript);
    using PointIndex = Edge::PointIndex;

    template <typename SourceT, typename ExtractorT = Subscript>
    requires Utilities::Concepts::ExtractorOf<ExtractorT, Point, SourceT, PointIndex>
    constexpr auto operator()(Edge edge
        , SourceT&& verticesSource
        , ExtractorT extractor = Utilities::subscript) const 
        noexcept(std::is_nothrow_invocable_v<ExtractorT, SourceT, PointIndex>)
    {
        Point a = std::invoke(extractor, verticesSource, edge[0]);
        Point b = std::invoke(extractor, verticesSource, edge[1]);

        return a.squaredDistance(b);
    }
};

template <>
struct MidpointImpl<Primitives::Edge>
{
public:

    using Point      = Primitives::Point;
    using Edge       = Primitives::Edge;
    using Subscript  = decltype(Utilities::subscript);
    using PointIndex = Edge::PointIndex;

    template <typename SourceT, typename ExtractorT = Subscript>
    requires Utilities::Concepts::ExtractorOf<ExtractorT, Point, SourceT, PointIndex>
    constexpr auto operator()(Edge edge
        , SourceT&& verticesSource
        , ExtractorT extractor = Utilities::subscript) const 
        noexcept(std::is_nothrow_invocable_v<ExtractorT, SourceT, PointIndex>)
    {
        Point a = std::invoke(extractor, verticesSource, edge[0]);
        Point b = std::invoke(extractor, verticesSource, edge[1]);

        return midpoint(a, b);
    }
};

template <>
struct CircumcircleImpl<Primitives::Edge>
{
public:

    using Point      = Primitives::Point;
    using Circle     = Primitives::Circle;
    using Edge       = Primitives::Edge;
    using Subscript  = decltype(Utilities::subscript);
    using PointIndex = Edge::PointIndex;

    template <typename SourceT, typename ExtractorT = Subscript>
    requires Utilities::Concepts::ExtractorOf<ExtractorT, Point, SourceT, PointIndex>
    constexpr auto operator()(Edge edge
        , SourceT&& verticesSource
        , ExtractorT extractor = Utilities::subscript) const 
        noexcept(std::is_nothrow_invocable_v<ExtractorT, SourceT, PointIndex>)
    {
        Point a = std::invoke(extractor, verticesSource, edge[0]);
        Point b = std::invoke(extractor, verticesSource, edge[1]);

        Point center = midpoint(a, b);

        return Circle{ center, center.squaredDistance(a) };
    }
};

} // namespace DelTriGen::Mesh::Tools

template <>
struct std::hash<DelTriGen::Mesh::Primitives::Edge>
{
public:

    using Edge         = DelTriGen::Mesh::Primitives::Edge;
    using PointIndex   = Edge::PointIndex;
    using PointIndexes = Edge::PointIndexes;

    std::size_t operator()(Edge edge) const noexcept
    {
        PointIndexes indexes = edge.indexes();

        return boost::hash_range(indexes.cbegin(), indexes.cend());
    }
};

#endif // DELTRIGEN_MESH_PRIMITIVES_EDGE_HPP
