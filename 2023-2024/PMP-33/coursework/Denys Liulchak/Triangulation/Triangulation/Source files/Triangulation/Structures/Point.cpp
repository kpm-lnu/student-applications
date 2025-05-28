#include <cmath>
#include <iostream>

#include "../../../Header files/Triangulation/Structures/Point.h"

namespace Triangulation::Structures
{
	Point::Point() noexcept 
		: x(0.0), y(0.0) {}

	Point::Point(double x, double y) noexcept 
		: x(x), y(y) {}

	void Point::swap(Point& other) noexcept
	{
		if (this == &other)
			return;

		using std::swap;

		swap(x, other.x);
		swap(y, other.y);
	}

	double Point::distance(const Point& other) const noexcept
	{
		double deltaX = other.x - x;
		double deltaY = other.y - y;

		double result = std::sqrt(deltaX * deltaX + deltaY * deltaY);

		return result;
	}

	bool Point::isOnLine(const Point& point1, const Point& point2) const noexcept
	{
		return (x - point1.x) / (point2.x - point1.x) ==
			   (y - point1.y) / (point2.y - point1.y);
	}

	bool Point::isOnLineSegment(const Point& point1, const Point& point2) const noexcept
	{
		if ((point1.x > x && point2.x > x) ||
			(point1.x < x && point2.x < x))
			return false;

		if ((point1.y > y && point2.y > y) ||
			(point1.y < y && point2.y < y))
			return false;

		return isOnLine(point1, point2);
	}

	Point& Point::operator+=(const Point& right) noexcept
	{
		x += right.x;
		y += right.y;

		return *this;
	}

	Point& Point::operator-=(const Point& right) noexcept
	{
		x -= right.x;
		y -= right.y;

		return *this;
	}

	Point& Point::operator*=(double scalar) noexcept
	{
		x *= scalar;
		y *= scalar;

		return *this;
	}

	Point& Point::operator/=(double scalar) noexcept
	{
		x /= scalar;
		y /= scalar;

		return *this;
	}

	Point operator+(const Point& left, const Point& right) noexcept
	{
		Point leftCopy(left);

		leftCopy += right;

		return leftCopy;
	}

	Point operator-(const Point& left, const Point& right) noexcept
	{
		Point leftCopy(left);

		leftCopy -= right;

		return leftCopy;
	}

	Point operator*(const Point& point, double scalar) noexcept
	{
		Point copyPoint(point);

		copyPoint *= scalar;

		return copyPoint;
	}

	Point operator*(double scalar, const Point& point) noexcept
	{
		return point * scalar;
	}

	Point operator/(const Point& point, double scalar) noexcept
	{
		Point copyPoint(point);

		copyPoint /= scalar;

		return copyPoint;
	}

	Point operator/(double scalar, const Point& point) noexcept
	{
		return point / scalar;
	}

	bool operator==(const Point& left, const Point& right) noexcept
	{
		constexpr double eps = 1E-9;

		return std::abs(left.x - right.x) < eps &&
			   std::abs(left.y - right.y) < eps;
	}

	bool operator!=(const Point& left, const Point& right) noexcept
	{
		return !(left == right);
	}

	std::istream& operator>>(std::istream& iStream, Point& point)
	{
		iStream >> point.x >> point.y;

		return iStream;
	}

	std::ostream& operator<<(std::ostream& oStream, const Point& point)
	{
		oStream << point.x << ' ' << point.y;

		return oStream;
	}
}
