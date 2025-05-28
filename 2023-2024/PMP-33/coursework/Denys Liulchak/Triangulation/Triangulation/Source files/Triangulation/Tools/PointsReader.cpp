#include <istream>

#include "../../../Header files/Triangulation/Tools/PointsReader.h"

using Triangulation::Structures::Points;

namespace Triangulation::Tools
{
	Points PointsReader::read(std::istream& iStream)
	{
		std::size_t size;

		iStream >> size;

		Points points(size);

		for (size_t i = 0; i < size; ++i)
			iStream >> points[i];

		return points;
	}
}
