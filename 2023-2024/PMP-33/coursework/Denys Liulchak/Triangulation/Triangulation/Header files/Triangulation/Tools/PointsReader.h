#pragma once

#include "../Structures/Points.h"

namespace Triangulation::Tools
{
	class PointsReader
	{
	private:
		using Points = Triangulation::Structures::Points;

	public:
		PointsReader() = delete;

		static Points read(std::istream& iStream);
	};
}
