#include "../../../Header files/Triangulation/Structures/Triangle.h"

namespace Triangulation::Structures
{
	Triangle::Triangle() noexcept
		: edge1(nullptr), edge2(nullptr), edge3(nullptr) {}

	Triangle::Triangle(const Edge* edge1, const Edge* edge2, const Edge* edge3) noexcept
		: edge1(edge1), edge2(edge2), edge3(edge3) {}
}
