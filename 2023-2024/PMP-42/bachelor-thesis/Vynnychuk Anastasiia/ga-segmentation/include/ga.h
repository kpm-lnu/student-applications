#include <vector>
#include "base.h"

void cross(Individual parent1, Individual parent2, unsigned char pos,
           Individual &child1, Individual &child2);

void mutate(unsigned int &ind, unsigned char mutation);

float rho(unsigned int x, unsigned int y, unsigned char gy0, unsigned char gy1,
          unsigned char gy2, const std::vector<std::vector<std::vector<unsigned char>>> &img);

float fitness(Individual ind, const std::vector<std::vector<std::vector<unsigned char>>> &img,
              std::vector<std::vector<float>> &segments);

std::vector<std::vector<float>>
get_segments_mvj(std::vector<Individual> population,
                 const std::vector<std::vector<std::vector<unsigned char>>> &img);
