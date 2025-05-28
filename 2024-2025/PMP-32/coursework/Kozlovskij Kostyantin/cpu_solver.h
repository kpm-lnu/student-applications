#pragma once
#include "matrix_generator.h"
#include <vector>
#include <stdexcept>

std::vector<double> conjugateGradientCPU(
    const CRSMatrix& A,
    const std::vector<double>& b,
    double tolerance,
    int max_iterations,
    int* iterations_used = nullptr,
    double* final_residual = nullptr
);

std::vector<double> matrixVectorMultiplyCRS(
    const CRSMatrix& A,
    const std::vector<double>& x
);

double dotProduct(const std::vector<double>& a, const std::vector<double>& b);

double computeResidualNorm(const CRSMatrix& A, const std::vector<double>& x, const std::vector<double>& b);
