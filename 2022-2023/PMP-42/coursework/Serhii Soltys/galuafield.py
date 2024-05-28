import math

class FiniteField:
    def __init__(self, value, modulus):
        if not isinstance(value, int) or not isinstance(modulus, int):
            raise TypeError("Both value and modulus must be integers")
        if not self._is_prime(modulus) and not self._is_prime_power(modulus):
            raise ValueError("Modulus must be a prime or prime power")
        self.value = value % modulus
        self.modulus = modulus

    def __repr__(self):
        return f"FiniteField({self.value}, {self.modulus})"

    def _phi(self, n):
        result = n
        p = 2
        while p * p <= n:
            if n % p == 0:
                while n % p == 0:
                    n = n // p
                result -= result // p
            p += 1
        if n > 1:
            result -= result // n
        return result

    def _is_prime(self, n):
        if n <= 1:
            return False
        return self._phi(n) == n - 1

    def _is_prime_power(self, n):
        if n <= 1:
            return False
        for i in range(2, int(math.sqrt(n)) + 1):
            if n % i == 0:
                while n % i == 0:
                    n = n // i
                if n != 1:
                    return False
                return self._is_prime(i)
        return self._is_prime(n)

    def __eq__(self, other):
        if isinstance(other, FiniteField):
            return self.value == other.value and self.modulus == other.modulus
        return False

    def __add__(self, other):
        if isinstance(other, FiniteField) and self.modulus == other.modulus:
            sum = (self.value + other.value) % self.modulus
            return FiniteField(sum, self.modulus)
        raise TypeError("Both operands must be instances of FiniteField class with the same modulus")

    def __sub__(self, other):
        if isinstance(other, FiniteField) and self.modulus == other.modulus:
            difference = (self.value - other.value) % self.modulus
            return FiniteField(difference, self.modulus)
        raise TypeError("Both operands must be instances of FiniteField class with the same modulus")

    def __mul__(self, other):
        if isinstance(other, FiniteField) and self.modulus == other.modulus:
            product = (self.value * other.value) % self.modulus
            return FiniteField(product, self.modulus)
        raise TypeError("Both operands must be instances of FiniteField class with the same modulus")

    def __truediv__(self, other):
        if isinstance(other, FiniteField) and self.modulus == other.modulus:
            reciprocal_value = self._find_reciprocal_element(other.value)
            quotient = (self.value * reciprocal_value) % self.modulus
            return FiniteField(quotient, self.modulus)
        raise TypeError("Both operands must be instances of FiniteField class with the same modulus")

    def __pow__(self, power):
        if isinstance(power, int):
            if power == 0:
                return FiniteField(1, self.modulus)
            elif power < 0:
                reciprocal_value = self._find_reciprocal_element(self.value)
                return FiniteField(reciprocal_value, self.modulus) ** abs(power)
            elif power % 2 == 0:
                half = self ** (power // 2)
                return half * half
            else:
                return self * (self ** (power - 1))
        raise TypeError("The exponent must be an integer")

    def _find_reciprocal_element(self, value):
        for i in range(1, self.modulus):
            if (value * i) % self.modulus == 1:
                return i
        raise ValueError(
            f"Cannot find reciprocal element for the value {value} in the finite field with modulus {self.modulus}")


a = FiniteField(5, 7)
b = FiniteField(3, 7)

print("A=", a)
print(a == b)

sum = a + b
print("Sum:", sum)

difference = a - b
print("Diff:", difference)

product = a * b
print("Mult:", product)

quotient = a / b
print("Div:", quotient)

power = a ** -3
print("A to power -3:", power)

power1 = b ** 4
print("B to power 4", power1)