#pragma once

#include <random>

#include "../Structures/Points.h"

namespace Triangulation::Tools 
{
	class ShuffleProvider
	{
	private:
		using Points = Triangulation::Structures::Points;

		std::mt19937 randomEngine;

	public:
		ShuffleProvider();

		void shuffle(Points& points, std::size_t offset);
	};
}
