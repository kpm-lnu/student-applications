#include <boost/math/special_functions/next.hpp>

#include <DelTriGen/FloatingPoint/Compare.hpp>

namespace DelTriGen::FloatingPoint
{

bool ulpEqual(double lhs, double rhs, double ulpTol) noexcept
{
    if (!std::isfinite(lhs) ||
        !std::isfinite(rhs))
    {
        return lhs == rhs;
    }

    return std::fabs(boost::math::float_distance(lhs, rhs)) <= ulpTol;
}

} // namespace DelTriGen::FloatingPoint
