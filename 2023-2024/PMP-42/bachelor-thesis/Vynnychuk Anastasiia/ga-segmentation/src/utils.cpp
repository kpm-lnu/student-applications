#include <vector>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "base.h"

int population_to_file(std::vector<Individual> population, unsigned int n)
{
    std::ofstream outputFile;

    outputFile.open("output.csv");

    if (!outputFile.is_open())
    {
        std::cerr << "Failed to open the file for writing!" << std::endl;
        return 1; // Return an error code
    }

    for (Individual ind : population)
    {
        unsigned char c = get_cluster(ind);
        unsigned int S2 = ind >> 8;
        unsigned int i = S2 / n;
        unsigned int j = S2 % n;

        outputFile << i << "," << j << "," << (unsigned int)c << std::endl;
    }

    return 0;
}

std::vector<std::vector<std::vector<uchar>>> mat_to_vec(cv::Mat img)
{
    int n = img.rows;
    int m = img.cols;
    int c = img.channels();

    std::vector<std::vector<std::vector<uchar>>> imgVec(n, std::vector<std::vector<uchar>>(m, std::vector<uchar>(c)));

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            for (int k = 0; k < c; k++)
            {
                imgVec[i][j][k] = img.at<cv::Vec3b>(i, j)[k];
            }
        }
    }

    return imgVec;
}
