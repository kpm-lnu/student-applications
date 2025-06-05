#ifndef DELTRIGEN_FLOATINGPOINT_COMPARE_HPP
#define DELTRIGEN_FLOATINGPOINT_COMPARE_HPP

#include <cmath>

namespace DelTriGen::FloatingPoint
{

constexpr bool absEqual(double lhs, double rhs, double absTol = 1E-13) noexcept;

constexpr bool relativeEqual(double lhs, double rhs, double relTol = 1E-13) noexcept;

constexpr bool almostEqual(double lhs, double rhs
    , double absTol = 1E-13
    , double relTol = 1E-13) noexcept;

bool ulpEqual(double lhs, double rhs, double ulpTol = 4) noexcept;

constexpr bool absEqual(double lhs, double rhs, double absTol) noexcept
{
    if (lhs == rhs)
    {
        return true;
    }

    return std::fabs(lhs - rhs) <= absTol;
}

constexpr bool relativeEqual(double lhs, double rhs, double relTol) noexcept
{
    if (!std::isfinite(lhs) ||
        !std::isfinite(rhs))
    {
        return lhs == rhs;
    }

    return std::fabs(lhs - rhs) <=
        relTol * std::fmax(std::fabs(lhs), std::fabs(rhs));
}

constexpr bool almostEqual(double lhs, double rhs
    , double absTol
    , double relTol) noexcept
{
    if (!std::isfinite(lhs) ||
        !std::isfinite(rhs))
    {
        return lhs == rhs;
    }

    return std::fabs(lhs - rhs) <=
        std::fmax(absTol, relTol * std::fmax(std::fabs(lhs), std::fabs(rhs)));
}

} // namespace DelTriGen::FloatingPoint

#endif // DELTRIGEN_FLOATINGPOINT_COMPARE_HPP
