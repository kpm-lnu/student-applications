#ifndef DELTRIGEN_MESH_PRIMITIVES_CIRCLE_HPP
#define DELTRIGEN_MESH_PRIMITIVES_CIRCLE_HPP

#include <numbers>

#include <DelTriGen/Mesh/Primitives/Point.hpp>
#include <DelTriGen/Mesh/Tools/Area.hpp>

namespace DelTriGen::Mesh::Primitives
{

class Circle
{
public:

    using CenterType     = Point;
    using CoordinateType = Point::CoordinateType;
    using RadiusType     = CoordinateType;

    constexpr Circle() noexcept;
    explicit constexpr Circle(RadiusType squaredRadius) noexcept;
    explicit constexpr Circle(Point center, RadiusType squaredRadius) noexcept;
    explicit constexpr Circle(CoordinateType centerX, CoordinateType centerY
        , RadiusType squaredRadius) noexcept;

    constexpr Point center() const noexcept;
    constexpr void center(Point value) noexcept;

    constexpr RadiusType radius() const noexcept;
    constexpr void radius(RadiusType value) noexcept;

    constexpr RadiusType squaredRadius() const noexcept;
    constexpr void squaredRadius(RadiusType value) noexcept;

    constexpr RadiusType circumference() const noexcept;

    constexpr RadiusType area() const noexcept;

    constexpr bool contains(Point point) const noexcept;

    constexpr bool isInside(Point point) const noexcept;
    constexpr bool isOnBoundary(Point point) const noexcept;
    constexpr bool isOutside(Point point) const noexcept;

    constexpr bool isAlmostOnBoundary(Point point
        , CoordinateType absTol = 1E-13
        , CoordinateType relTol = 1E-13) const noexcept;

    constexpr bool almostEqual(const Circle& other
        , CoordinateType absTol = 1E-13
        , CoordinateType relTol = 1E-13) const noexcept;

    constexpr auto operator<=>(const Circle&) const noexcept = default;

    friend std::istream& operator>>(std::istream& is, Circle& circle);
    friend std::ostream& operator<<(std::ostream& os, const Circle& circle);

private:

    Point center_;
    RadiusType squaredRadius_;
};



constexpr Circle::Circle() noexcept
    : center_{}, squaredRadius_{}
{}

constexpr Circle::Circle(RadiusType squaredRadius) noexcept
    : center_{}, squaredRadius_{ squaredRadius }
{}

constexpr Circle::Circle(Point center, RadiusType squaredRadius) noexcept
    : center_{ center }, squaredRadius_{ squaredRadius }
{}

constexpr Circle::Circle(CoordinateType centerX, CoordinateType centerY
    , RadiusType squaredRadius) noexcept
    : center_{ centerX, centerY }, squaredRadius_{ squaredRadius }
{}

constexpr Point Circle::center() const noexcept
{
    return center_;
}

constexpr void Circle::center(Point value) noexcept
{
    center_ = value;
}

constexpr Circle::RadiusType Circle::radius() const noexcept
{
    return std::sqrt(squaredRadius_);
}

constexpr void Circle::radius(RadiusType value) noexcept
{
    squaredRadius_ = value * value;
}

constexpr Circle::RadiusType Circle::squaredRadius() const noexcept
{
    return squaredRadius_;
}

constexpr void Circle::squaredRadius(RadiusType value) noexcept
{
    squaredRadius_ = value;
}

constexpr Circle::RadiusType Circle::circumference() const noexcept
{
    return 2 * std::numbers::pi_v<RadiusType> * radius();
}

constexpr Circle::RadiusType Circle::area() const noexcept
{
    return std::numbers::pi_v<RadiusType> * squaredRadius_;
}

constexpr bool Circle::contains(Point point) const noexcept
{
    return center_.squaredDistance(point) <= squaredRadius_;
}

constexpr bool Circle::isInside(Point point) const noexcept
{
    return center_.squaredDistance(point) < squaredRadius_;
}

constexpr bool Circle::isOnBoundary(Point point) const noexcept
{
    return center_.squaredDistance(point) == squaredRadius_;
}

constexpr bool Circle::isOutside(Point point) const noexcept
{
    return center_.squaredDistance(point) > squaredRadius_;
}

constexpr bool Circle::isAlmostOnBoundary(Point point
    , CoordinateType absTol
    , CoordinateType relTol) const noexcept
{
    return FloatingPoint::almostEqual(center_.squaredDistance(point), squaredRadius_,
        absTol, relTol);
}

constexpr bool Circle::almostEqual(const Circle& other
    , CoordinateType absTol
    , CoordinateType relTol) const noexcept
{
    return center_.almostEqual(other.center_, absTol, relTol) &&
        FloatingPoint::almostEqual(squaredRadius_, other.squaredRadius_, absTol, relTol);
}

} // namespace DelTriGen::Mesh::Primitives

namespace DelTriGen::Mesh::Tools
{

template <>
struct AreaImpl<Primitives::Circle>
{
public:

    using Circle = Primitives::Circle;

    constexpr auto operator()(const Circle& circle) const noexcept
    {
        return circle.area();
    }
};

} // namespace DelTriGen::Mesh::Tools

template <>
struct std::hash<DelTriGen::Mesh::Primitives::Circle>
{
public:

    using Circle         = DelTriGen::Mesh::Primitives::Circle;
    using CenterType     = Circle::CenterType;
    using CoordinateType = Circle::CoordinateType;
    using RadiusType     = Circle::RadiusType;

    std::size_t operator()(const Circle& circle) const noexcept
    {
        std::hash<CenterType> hasher;
        std::size_t seed = hasher(circle.center());

        boost::hash_combine(seed, circle.squaredRadius());
        
        return seed;
    }
};

#endif // DELTRIGEN_MESH_PRIMITIVES_CIRCLE_HPP
