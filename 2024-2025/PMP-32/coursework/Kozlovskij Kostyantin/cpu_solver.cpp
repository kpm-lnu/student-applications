#include "cpu_solver.h"
#include <cmath>
#include <iostream>
#include <chrono>

std::vector<double> matrixVectorMultiplyCRS(const CRSMatrix& A, const std::vector<double>& x) {
    if (A.cols != x.size()) {
        throw std::invalid_argument("Matrix column count must match the size of vector x.");
    }

    std::vector<double> result(A.rows, 0.0);
    for (int i = 0; i < A.rows; ++i) {
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            result[i] += A.values[j] * x[A.col_indices[j]];
        }
    }
    return result;
}

double dotProduct(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must be the same size.");
    }

    double result = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}

double computeResidualNorm(const CRSMatrix& A, const std::vector<double>& x, const std::vector<double>& b) {
    std::vector<double> Ax = matrixVectorMultiplyCRS(A, x);
    double norm = 0.0;
    for (size_t i = 0; i < b.size(); ++i) {
        double diff = Ax[i] - b[i];
        norm += diff * diff;
    }
    return std::sqrt(norm);
}

std::vector<double> conjugateGradientCPU(
    const CRSMatrix& A,
    const std::vector<double>& b,
    double tolerance,
    int max_iterations,
    int* iterations_used,
    double* final_residual
) {
    if (A.rows != A.cols) {
        throw std::invalid_argument("Matrix must be square.");
    }
    if (A.rows != b.size()) {
        throw std::invalid_argument("Size of b must match matrix dimensions.");
    }

    auto start = std::chrono::high_resolution_clock::now();
    int n = A.rows;

    std::vector<double> precond(n);
    for (int i = 0; i < n; ++i) {
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            if (A.col_indices[j] == i) {
                precond[i] = 1.0 / A.values[j];
                break;
            }
        }
    }

    std::vector<double> x(n, 0.0);
    std::vector<double> r = b;

    std::vector<double> z(n);
    for (int i = 0; i < n; ++i) {
        z[i] = precond[i] * r[i];
    }

    std::vector<double> p = z;

    double rz = dotProduct(r, z);
    double init_residual = std::sqrt(dotProduct(r, r));
    std::cout << "CPU Solver: Initial residual = " << init_residual << std::endl;

    int iter;
    double residual;

    for (iter = 0; iter < max_iterations; ++iter) {
        std::vector<double> Ap = matrixVectorMultiplyCRS(A, p);
        double pAp = dotProduct(p, Ap);

        if (pAp <= 1e-14) {
            std::cerr << "CPU Solver: Breakdown, p^T * A * p ? 0." << std::endl;
            break;
        }

        double alpha = rz / pAp;

        for (int i = 0; i < n; ++i) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }

        residual = std::sqrt(dotProduct(r, r));

        if (residual < tolerance) {
            if (iterations_used) *iterations_used = iter + 1;
            if (final_residual) *final_residual = residual;
            break;
        }

        for (int i = 0; i < n; ++i) {
            z[i] = precond[i] * r[i];
        }

        double rz_new = dotProduct(r, z);
        double beta = rz_new / rz;

        for (int i = 0; i < n; ++i) {
            p[i] = z[i] + beta * p[i];
        }

        rz = rz_new;

        if ((iter + 1) % 100 == 0) {
            std::cout << "CPU Solver: Iteration " << (iter + 1)
                << ", Residual = " << residual << std::endl;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    if (iter == max_iterations) {
        std::cout << "CPU Solver: Maximum iterations reached without convergence." << std::endl;
        if (iterations_used) *iterations_used = max_iterations;
        if (final_residual) *final_residual = residual;
    }
    else {
        std::cout << "CPU Solver: Converged in " << (iter + 1) << " iterations." << std::endl;
    }
    std::cout << "CPU Solver: Elapsed time = " << elapsed.count() << " seconds." << std::endl;
    std::cout << "CPU Solver: Final residual norm = " << computeResidualNorm(A, x, b) << std::endl;

    return x;
}
