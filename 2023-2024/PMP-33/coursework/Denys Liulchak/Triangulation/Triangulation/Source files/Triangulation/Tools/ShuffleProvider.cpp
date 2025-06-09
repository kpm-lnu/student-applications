#include <random>

#include "../../../Header files/Triangulation/Tools/ShuffleProvider.h"

namespace Triangulation::Tools
{
	ShuffleProvider::ShuffleProvider()
		: randomEngine(std::random_device()()) {}

	void ShuffleProvider::shuffle(Points& points, std::size_t offset)
	{
        std::shuffle(points.begin() + offset, points.end(), randomEngine);
	}
}
