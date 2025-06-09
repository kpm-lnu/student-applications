#pragma once

#include "../Structures/Edge.h"
#include "../Structures/Point.h"

namespace Triangulation::Tools
{
	class DelaunayCondition
	{
	private:
		using Edge  = Triangulation::Structures::Edge;
		using Point = Triangulation::Structures::Point;

	public:
		static bool isSatisfied(const Edge& edge, const Point& point0, const Point& point2);
	};
}