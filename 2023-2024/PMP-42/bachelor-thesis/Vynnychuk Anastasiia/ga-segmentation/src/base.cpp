#include "base.h"

unsigned char get_cluster(Individual &ind)
{
	float denominator = 255 / K;
	unsigned int cluster = (ind & 255U);
	float result = cluster / denominator;
	result = result >= K ? K - 1 : result;
	ind = (ind & (~255U)) | ((unsigned int)(denominator * result));
	return result;
}
