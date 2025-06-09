#include "../../../Header files/Triangulation/Structures/Edge.h"

namespace Triangulation::Structures
{
	Edge::Edge() noexcept
		: point1(nullptr), point2(nullptr)
	    , triangle1(nullptr), triangle2(nullptr) {}

	Edge::Edge(const Point* point1, const Point* point2) noexcept 
		: point1(point1), point2(point2)
	    , triangle1(nullptr), triangle2(nullptr) {}

	Edge::Edge(const Point* point1, const Point* point2,
		       const Triangle* triangle1, const Triangle* triangle2) noexcept
		: point1(point1), point2(point2)
		, triangle1(triangle1), triangle2(triangle2) {}
}
