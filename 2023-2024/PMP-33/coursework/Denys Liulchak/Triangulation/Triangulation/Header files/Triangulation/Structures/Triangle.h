#pragma once

namespace Triangulation::Structures
{
	struct Edge;

	struct Triangle
	{
	public:
		const Edge* edge1;
		const Edge* edge2;
		const Edge* edge3;

		Triangle() noexcept;
		explicit Triangle(const Edge* edge1, const Edge* edge2, const Edge* edge3) noexcept;
	};
}
