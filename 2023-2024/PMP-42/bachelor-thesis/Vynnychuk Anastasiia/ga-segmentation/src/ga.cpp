#include <vector>
#include "base.h"

void cross(Individual parent1, Individual parent2, unsigned char pos,
		   Individual &child1, Individual &child2)
{
	child1 = (parent1 & (~0U << (32 - pos))) | (parent2 & (~0U >> pos));
	child2 = (parent1 & (~0U >> pos)) | (parent2 & (~0U << (32 - pos)));
}

void mutate(unsigned int &ind, unsigned char mutation) { ind ^= mutation; }

float rho(unsigned int x, unsigned int y, float gy0, float gy1,
		  float gy2, const std::vector<std::vector<std::vector<unsigned char>>> &img)
{
	unsigned int h = img.size();
	unsigned int w = img[0].size();

	// float d1 = (float)std::abs(img[x][y][0] - gy0) /
	//	std::max<unsigned char>(img[x][y][0], gy0);
	float d1 = (float)std::abs(img[x][y][0] - gy0) /
				   std::max<float>(img[x][y][0], gy0) +
			   (float)std::abs(img[x][y][1] - gy1) /
				   std::max<float>(img[x][y][1], gy1) +
			   (float)std::abs(img[x][y][2] - gy2) /
				   std::max<float>(img[x][y][2], gy2);

	float d2 = 0;

	for (int i = x < R ? 0 : x - R; i < (x >= h - R ? h : x + R + 1); i++)
	{
		for (int j = y < R ? 0 : y - R; j < (y >= w - R ? w : y + R + 1); j++)
		{
			// d2 += (float)std::abs(img[i][j][0] - gy0) /
			//	std::max<unsigned char>(img[i][j][0], gy0);

			d2 += (float)std::abs(img[i][j][0] - gy0) /
					  std::max<float>(img[i][j][0], gy0) +
				  (float)std::abs(img[i][j][1] - gy1) /
					  std::max<float>(img[i][j][1], gy1) +
				  (float)std::abs(img[i][j][2] - gy2) /
					  std::max<float>(img[i][j][2], gy2);
		}
	}

	return A * d1 + B * d2;
}

float fitness(Individual ind, const std::vector<std::vector<std::vector<unsigned char>>> &img,
			  std::vector<std::vector<float>> &segments)
{
	unsigned int m = img[0].size();
	unsigned char c = get_cluster(ind);
	unsigned int pos = ind >> 8;
	unsigned int i = pos / m;
	unsigned int j = pos % m;

	return rho(i, j, segments[c][0], segments[c][1], segments[c][2], img);
}

std::vector<std::vector<float>>
get_segments_mvj(std::vector<Individual> population,
				 const std::vector<std::vector<std::vector<unsigned char>>> &img)
{
	unsigned int m = img[0].size();

	std::vector<std::vector<float>> segments(K, std::vector<float>(3, 0));
	std::vector<int> counts(K, 0);

	unsigned char cluster;
	unsigned int pos, i, j;

	for (Individual ind : population)
	{
		cluster = get_cluster(ind);
		pos = ind >> 8;
		i = pos / m;
		j = pos % m;
		++counts[cluster];
		segments[cluster][0] =
			segments[cluster][0] + (img[i][j][0] - segments[cluster][0]) / (float)counts[cluster];
		segments[cluster][1] =
			segments[cluster][1] + (img[i][j][1] - segments[cluster][1]) / (float)counts[cluster];
		segments[cluster][2] =
			segments[cluster][2] + (img[i][j][2] - segments[cluster][2]) / (float)counts[cluster];
	}

	return segments;
}
