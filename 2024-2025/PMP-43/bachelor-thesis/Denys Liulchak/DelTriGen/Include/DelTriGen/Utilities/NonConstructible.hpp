#ifndef DELTRIGEN_UTILITIES_NONCONSTRUCTIBLE_HPP
#define DELTRIGEN_UTILITIES_NONCONSTRUCTIBLE_HPP

namespace DelTriGen::Utilities
{

struct NonConstructible
{
public:

    NonConstructible()                                   = delete;
    NonConstructible(const NonConstructible&)            = delete;
    NonConstructible(NonConstructible&&)                 = delete;

    NonConstructible& operator=(const NonConstructible&) = delete;
    NonConstructible& operator=(NonConstructible&&)      = delete;
};

} // namespace DelTriGen::Utilities

#endif // DELTRIGEN_UTILITIES_NONCONSTRUCTIBLE_HPP
