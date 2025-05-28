#pragma once

#include <iosfwd>

namespace Triangulation::Structures
{
	struct Point
	{
	public:
		double x;
		double y;

		Point() noexcept;
		explicit Point(double x, double y) noexcept;

		void swap(Point& other) noexcept;

		double distance(const Point& other) const noexcept;

		bool isOnLine(const Point& point1, const Point& point2) const noexcept;
		bool isOnLineSegment(const Point& point1, const Point& point2) const noexcept;

		Point& operator+=(const Point& right) noexcept;
		Point& operator-=(const Point& right) noexcept;

		Point& operator*=(double scalar) noexcept;
		Point& operator/=(double scalar) noexcept;
	};

	Point operator+(const Point& left, const Point& right) noexcept;
	Point operator-(const Point& left, const Point& right) noexcept;

	Point operator*(const Point& point, double scalar) noexcept;
	Point operator*(double scalar, const Point& point) noexcept;

	Point operator/(const Point& point, double scalar) noexcept;
	Point operator/(double scalar, const Point& point) noexcept;

	bool operator==(const Point& left, const Point& right) noexcept;
	bool operator!=(const Point& left, const Point& right) noexcept;

	std::istream& operator>>(std::istream& iStream, Point& point);
	std::ostream& operator<<(std::ostream& oStream, const Point& point);
}
