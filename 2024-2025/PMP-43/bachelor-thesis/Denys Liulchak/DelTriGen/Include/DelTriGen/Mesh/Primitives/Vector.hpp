#ifndef DELTRIGEN_MESH_PRIMITIVES_VECTOR_HPP
#define DELTRIGEN_MESH_PRIMITIVES_VECTOR_HPP

#include <cstddef>
#include <iosfwd>
#include <array>
#include <numeric>
#include <algorithm>
#include <functional>

#include <boost/functional/hash.hpp>

#include <DelTriGen/FloatingPoint/Compare.hpp>
#include <DelTriGen/Mesh/Tools/Length.hpp>
#include <DelTriGen/Mesh/Tools/SquaredLength.hpp>

namespace DelTriGen::Mesh::Primitives
{

class Vector
{
public:

    static constexpr std::size_t ndims = 2;

    using CoordinateType = double;
    using Coordinates    = std::array<CoordinateType, ndims>;

    constexpr Vector() noexcept;
    explicit constexpr Vector(CoordinateType value) noexcept;
    explicit constexpr Vector(CoordinateType x, CoordinateType y) noexcept;

    constexpr Coordinates coordinates() const noexcept;

    constexpr CoordinateType length() const noexcept;
    constexpr CoordinateType squaredLength() const noexcept;

    constexpr CoordinateType norm() const noexcept;
    constexpr CoordinateType squaredNorm() const noexcept;
    constexpr void normalize() noexcept;

    constexpr CoordinateType distance(Vector other) const noexcept;
    constexpr CoordinateType squaredDistance(Vector other) const noexcept;

    constexpr CoordinateType innerProduct(Vector other) const noexcept;
    constexpr CoordinateType crossProduct(Vector other) const noexcept;

    constexpr bool almostEqual(Vector other
        , CoordinateType absTol = 1E-13
        , CoordinateType relTol = 1E-13) const noexcept;

    constexpr CoordinateType& operator[](std::size_t index) noexcept;
    constexpr const CoordinateType& operator[](std::size_t index) const noexcept;

    constexpr Vector& operator=(Coordinates rhs) & noexcept;

    constexpr Vector& operator+=(Vector rhs) & noexcept;
    constexpr Vector& operator-=(Vector rhs) & noexcept;
    constexpr Vector& operator*=(Vector rhs) & noexcept;
    constexpr Vector& operator/=(Vector rhs) & noexcept;
    
    constexpr Vector& operator+=(CoordinateType scalar) & noexcept;
    constexpr Vector& operator-=(CoordinateType scalar) & noexcept;
    constexpr Vector& operator*=(CoordinateType scalar) & noexcept;
    constexpr Vector& operator/=(CoordinateType scalar) & noexcept;

    constexpr auto operator<=>(const Vector&) const noexcept = default;

    friend std::istream& operator>>(std::istream& is, Vector& vector);
    friend std::ostream& operator<<(std::ostream& os, Vector vector);

private:

    Coordinates coordinates_;
};

constexpr Vector operator+(Vector lhs, Vector rhs) noexcept;
constexpr Vector operator-(Vector lhs, Vector rhs) noexcept;
constexpr Vector operator*(Vector lhs, Vector rhs) noexcept;
constexpr Vector operator/(Vector lhs, Vector rhs) noexcept;

constexpr Vector operator+(Vector vector, Vector::CoordinateType scalar) noexcept;
constexpr Vector operator+(Vector::CoordinateType scalar, Vector vector) noexcept;

constexpr Vector operator-(Vector vector, Vector::CoordinateType scalar) noexcept;
constexpr Vector operator-(Vector::CoordinateType scalar, Vector vector) noexcept;

constexpr Vector operator*(Vector vector, Vector::CoordinateType scalar) noexcept;
constexpr Vector operator*(Vector::CoordinateType scalar, Vector vector) noexcept;

constexpr Vector operator/(Vector vector, Vector::CoordinateType scalar) noexcept;
constexpr Vector operator/(Vector::CoordinateType scalar, Vector vector) noexcept;



constexpr Vector::Vector() noexcept
    : coordinates_{}
{}

constexpr Vector::Vector(CoordinateType value) noexcept
{
    coordinates_.fill(value);
}

constexpr Vector::Vector(CoordinateType x, CoordinateType y) noexcept
    : coordinates_{ x, y }
{}

constexpr Vector::Coordinates Vector::coordinates() const noexcept
{
    return coordinates_;
}

constexpr Vector::CoordinateType Vector::length() const noexcept
{
    return std::sqrt(squaredLength());
}

constexpr Vector::CoordinateType Vector::squaredLength() const noexcept
{
    return std::inner_product(coordinates_.cbegin(), coordinates_.cend(),
        coordinates_.cbegin(), CoordinateType{});
}

constexpr Vector::CoordinateType Vector::norm() const noexcept
{
    return length();
}

constexpr Vector::CoordinateType Vector::squaredNorm() const noexcept
{
    return squaredLength();
}

constexpr void Vector::normalize() noexcept
{
    *this /= length();
}

constexpr Vector::CoordinateType Vector::distance(Vector other) const noexcept
{
    return std::sqrt(squaredDistance(other));
}

constexpr Vector::CoordinateType Vector::squaredDistance(Vector other) const noexcept
{
    return (other -= *this).squaredLength();
}

constexpr Vector::CoordinateType Vector::innerProduct(Vector other) const noexcept
{
    return std::inner_product(coordinates_.cbegin(), coordinates_.cend(),
        other.coordinates_.cbegin(), CoordinateType{});
}

constexpr Vector::CoordinateType Vector::crossProduct(Vector other) const noexcept
{
    auto& [x1, y1] = coordinates_;
    auto& [x2, y2] = other.coordinates_;

    return x1 * y2 - y1 * x2;
}

constexpr bool Vector::almostEqual(Vector other
    , CoordinateType absTol
    , CoordinateType relTol) const noexcept
{
    return std::ranges::equal(coordinates_, other.coordinates_,
        [absTol, relTol](CoordinateType lhs, CoordinateType rhs) noexcept
        {
            return  FloatingPoint::almostEqual(lhs, rhs, absTol, relTol);
        });
}

constexpr Vector::CoordinateType& Vector::operator[](std::size_t index) noexcept
{
    return coordinates_[index];
}

constexpr const Vector::CoordinateType& Vector::operator[](std::size_t index)
    const noexcept
{
    return coordinates_[index];
}

constexpr Vector& Vector::operator=(Coordinates rhs) & noexcept
{
    coordinates_ = rhs;

    return *this;
}

constexpr Vector& Vector::operator+=(Vector rhs) & noexcept
{
    std::ranges::transform(coordinates_, rhs.coordinates_, coordinates_.begin(),
        std::plus{});

    return *this;
}

constexpr Vector& Vector::operator-=(Vector rhs) & noexcept
{
    std::ranges::transform(coordinates_, rhs.coordinates_, coordinates_.begin(),
        std::minus{});

    return *this;
}

constexpr Vector& Vector::operator*=(Vector rhs) & noexcept
{
    std::ranges::transform(coordinates_, rhs.coordinates_, coordinates_.begin(),
        std::multiplies{});

    return *this;
}

constexpr Vector& Vector::operator/=(Vector rhs) & noexcept
{
    std::ranges::transform(coordinates_, rhs.coordinates_, coordinates_.begin(),
        std::divides{});

    return *this;
}

constexpr Vector& Vector::operator+=(CoordinateType scalar) & noexcept
{
    std::ranges::for_each(coordinates_,
        [scalar](CoordinateType& coordinate) noexcept
        {
            coordinate += scalar;
        });

    return *this;
}

constexpr Vector& Vector::operator-=(CoordinateType scalar) & noexcept
{
    std::ranges::for_each(coordinates_,
        [scalar](CoordinateType& coordinate) noexcept
        {
            coordinate -= scalar;
        });

    return *this;
}

constexpr Vector& Vector::operator*=(CoordinateType scalar) & noexcept
{
    std::ranges::for_each(coordinates_,
        [scalar](CoordinateType& coordinate) noexcept
        {
            coordinate *= scalar;
        });

    return *this;
}

constexpr Vector& Vector::operator/=(CoordinateType scalar) & noexcept
{
    std::ranges::for_each(coordinates_,
        [scalar](CoordinateType& coordinate) noexcept
        {
            coordinate /= scalar;
        });

    return *this;
}

constexpr Vector operator+(Vector lhs, Vector rhs) noexcept
{
    lhs += rhs;

    return lhs;
}

constexpr Vector operator-(Vector lhs, Vector rhs) noexcept
{
    lhs -= rhs;

    return lhs;
}

constexpr Vector operator*(Vector lhs, Vector rhs) noexcept
{
    lhs *= rhs;

    return lhs;
}

constexpr Vector operator/(Vector lhs, Vector rhs) noexcept
{
    lhs /= rhs;

    return lhs;
}

constexpr Vector operator+(Vector vector, Vector::CoordinateType scalar) noexcept
{
    vector += scalar;

    return vector;
}

constexpr Vector operator+(Vector::CoordinateType scalar, Vector vector) noexcept
{
    return vector + scalar;
}

constexpr Vector operator-(Vector vector, Vector::CoordinateType scalar) noexcept
{
    vector -= scalar;

    return vector;
}

constexpr Vector operator-(Vector::CoordinateType scalar, Vector vector) noexcept
{
    Vector result(scalar);

    result -= vector;

    return result;
}

constexpr Vector operator*(Vector vector, Vector::CoordinateType scalar) noexcept
{
    vector *= scalar;

    return vector;
}

constexpr Vector operator*(Vector::CoordinateType scalar, Vector vector) noexcept
{
    return vector * scalar;
}

constexpr Vector operator/(Vector vector, Vector::CoordinateType scalar) noexcept
{
    vector /= scalar;

    return vector;
}

constexpr Vector operator/(Vector::CoordinateType scalar, Vector vector) noexcept
{
    Vector result(scalar);

    result /= vector;

    return result;
}

} // namespace DelTriGen::Mesh::Primitives

namespace DelTriGen::Mesh::Tools
{

template <>
struct LengthImpl<Primitives::Vector>
{
public:

    using Vector = Primitives::Vector;

    constexpr auto operator()(Vector vector) const noexcept
    {
        return vector.length();
    }
};

template <>
struct SquaredLengthImpl<Primitives::Vector>
{
public:

    using Vector = Primitives::Vector;

    constexpr auto operator()(Vector vector) const noexcept
    {
        return vector.squaredLength();
    }
};

} // namespace DelTriGen::Mesh::Tools

template <>
struct std::hash<DelTriGen::Mesh::Primitives::Vector>
{
public:

    using Vector         = DelTriGen::Mesh::Primitives::Vector;
    using CoordinateType = Vector::CoordinateType;
    using Coordinates    = Vector::Coordinates;

    std::size_t operator()(Vector vector) const noexcept
    {
        Coordinates coordinates = vector.coordinates();

        return boost::hash_range(coordinates.cbegin(), coordinates.cend());
    }
};

#endif // DELTRIGEN_MESH_PRIMITIVES_VECTOR_HPP
