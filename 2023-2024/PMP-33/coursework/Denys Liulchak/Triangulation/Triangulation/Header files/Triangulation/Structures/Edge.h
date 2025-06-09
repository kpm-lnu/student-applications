#pragma once

namespace Triangulation::Structures
{
	struct Point;
	struct Triangle;

	struct Edge
	{
	public:
		const Point* point1;
		const Point* point2;

		const Triangle* triangle1;
		const Triangle* triangle2;

		Edge() noexcept;
		explicit Edge(const Point* point1, const Point* point2) noexcept;
		explicit Edge(const Point* point1, const Point* point2,
			          const Triangle* triangle1, const Triangle* triangle2) noexcept;
	};
}
