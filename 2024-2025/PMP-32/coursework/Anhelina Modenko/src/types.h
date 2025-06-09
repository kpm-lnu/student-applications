//
// Created by anmode on 28.06.2024.
//

#ifndef FEM_ENGINE_TYPES_H
#define FEM_ENGINE_TYPES_H

#include <string>
#include <vector>

/**
 * @enum Type
 * @brief Enumerates possible types of elements.
 */
enum class Type : unsigned char {
    SCALAR = 0,
    VEC2,
    VEC3,
    VEC4,
    ARRAY,
    STRUCT,
    INVALID
};

/**
 * @brief Rounds up a value to the next multiple of 2^n.
 *
 * @relation utility function
 * @param val The value to round up.
 * @param n The power of 2.
 * @return The rounded-up value.
 */
inline unsigned int round_up_pow2(unsigned int val, unsigned char n) {
    unsigned int pow2n = 0b1 << n; // = 1 * 2^n = 2^n
    unsigned int divisor = pow2n - 1; // = 0b0111...111 (n 1s)

    // last n bits = remainder of val / 2^n
    // add (2^n - rem) to get to the next multiple of 2^n
    unsigned int rem = val & divisor;
    if (rem) {
        val += pow2n - rem;
    }

    return val;
}

#endif //FEM_ENGINE_TYPES_H
