#include <iostream>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <random>
#include "utils.h"
#include "ga.h"
#include <string>

unsigned int R = 1;
unsigned char K = 4;
float A = .8f;
float B = .2f;
float P_C = .8f;
float P_M = .01f;
unsigned int N_ITER = 300;

int main(int argc, char *argv[])
{
    if (argc < 8)
    {
        std::cerr << "You should supply 6 args: K, A, B, P_C, P_M, N_ITER" << std::endl;
        return 1;
    }

    K = (unsigned char)std::stoi(argv[2]);
    A = std::stof(argv[3]);
    B = std::stof(argv[4]);
    P_C = std::stof(argv[5]);
    P_M = std::stof(argv[6]);
    N_ITER = std::stoi(argv[7]);

    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);
    cv::Mat mat = cv::imread(argv[1], cv::IMREAD_COLOR);

    float scaleFactor = argc > 8 ? std::stof(argv[8]) : .3f;
    cv::Mat rescaled;
    cv::resize(mat, rescaled, cv::Size(), scaleFactor, scaleFactor, cv::INTER_LINEAR);

    unsigned int n = rescaled.rows;
    unsigned int m = rescaled.cols;

    std::cout << n << ' ' << m << std::endl;

    std::vector<std::vector<std::vector<uchar>>> img = mat_to_vec(rescaled);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, K - 1);

    std::uniform_int_distribution<> pos_dis(25, 31);
    std::uniform_real_distribution<> mut_dis(0, 1);

    std::vector<Individual> population(n * m);

    for (unsigned int k = 0; k < n * m; k++)
    {
        uchar L = (255 / K) * dis(gen);
        Individual ind = (k << 8) | L;

        population[k] = ind;
    }

    uchar mutation;
    Individual child1, child2;
    std::vector<Individual> new_population(population.size());

    for (int iter = 0; iter < N_ITER; iter++)
    {
        std::vector<std::vector<float>> segments = get_segments_mvj(population, img);

        for (unsigned int i = 0; i < n * m / 2; i++)
        {
            child1 = population[i];
            child2 = population[n * m - 1U - i];

            if (mut_dis(gen) < P_C)
            {
                uchar pos = pos_dis(gen);
                cross(population[i], population[n * m - 1U - i],
                      pos, child1, child2);
            }

            mutation = 0;
            for (int i = 0; i < 8; ++i)
            {
                double r = mut_dis(gen);
                if (r < P_M)
                    mutation |= (1 << i);
            }
            mutate(child1, mutation);

            mutation = 0;
            for (int i = 0; i < 8; ++i)
            {
                double r = mut_dis(gen);
                if (r < P_M)
                    mutation |= (1 << i);
            }
            mutate(child2, mutation);

            new_population[i] = child1;
            new_population[n * m - 1 - i] = child2;
        }

        std::vector<std::vector<float>> new_segments = get_segments_mvj(new_population, img);

        for (unsigned int i = 0; i < n * m; i++)
        {
            if (fitness(population[i], img, segments) >
                fitness(new_population[i], img, new_segments))
            {
                population[i] = new_population[i];
                // segments = get_segments_mvj(population, img);
            }
        }

        // std::cout << "ended iteration " << iter + 1 << std::endl;
    }

    population_to_file(population, m);

    std::vector<uchar> resvec(n * m);

    for (unsigned int i = 0; i < n * m; i++)
    {
        resvec[i] = 255 / K * get_cluster(population[i]);
        // resvec[i] = get_cluster(population[i]);
    }

    cv::Mat result(resvec);
    cv::Mat resultSmoothed;
    result = result.reshape(1, n);
    cv::medianBlur(result, resultSmoothed, 3);

    cv::imwrite("./out.jpg", result);
    cv::imwrite("./out_smoothed.jpg", resultSmoothed);

    cv::imshow("orig", rescaled);
    cv::imshow("result", result);
    cv::imshow("result smoothed", resultSmoothed);
    cv::waitKey(0);

    return 0;
}
