#include <vector>
#include <opencv2/opencv.hpp>
#include "base.h"

int population_to_file(std::vector<Individual> population, unsigned int n);

std::vector<std::vector<std::vector<uchar>>> mat_to_vec(cv::Mat img);
