#include "matrix_generator.h"
#include "cpu_solver.h"
#include "gpu_solver.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <string>
#include <fstream>

double vectorNorm(const std::vector<double>& vec) {
    double sum = 0.0;
    for (double val : vec) {
        sum += val * val;
    }
    return std::sqrt(sum);
}

std::vector<double> vectorDifference(const std::vector<double>& a, const std::vector<double>& b) {
    std::vector<double> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] - b[i];
    }
    return result;
}

double relativeError(const std::vector<double>& a, const std::vector<double>& b) {
    std::vector<double> diff = vectorDifference(a, b);
    return vectorNorm(diff) / vectorNorm(a);
}

bool verifySolution(const CRSMatrix& A, const std::vector<double>& x, const std::vector<double>& b) {
    std::vector<double> Ax = matrixVectorMultiplyCRS(A, x);
    std::vector<double> residual = vectorDifference(Ax, b);
    double error = vectorNorm(residual) / vectorNorm(b);

    std::cout << "Solution verification: ||Ax - b|| / ||b|| = " << error << std::endl;

    return error < 1e-6;
}

int main() {
    std::cout << "===== Sparse Linear System Solver Benchmark =====" << std::endl;

    int matrix_size = 10000;
    int num_diagonals = 25;
    double tolerance = 1e-6;
    int max_iterations = 5000;
    std::string matrix_file_prefix = "matrix";
    std::string rhs_file = "rhs.txt";

    std::cout << "Enter matrix size (default: 10000): ";
    std::string input;
    std::getline(std::cin, input);
    if (!input.empty()) {
        matrix_size = std::stoi(input);
    }

    std::cout << "Enter number of diagonals (default: 25): ";
    input.clear();
    std::getline(std::cin, input);
    if (!input.empty()) {
        num_diagonals = std::stoi(input);
    }

    std::cout << "\nStep 1: Generating symmetric matrix with " << num_diagonals
        << " diagonals of size " << matrix_size << "x" << matrix_size << "..." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    CRSMatrix A = generateSymmetricDiagonalMatrix(matrix_size, num_diagonals);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Matrix generated with " << A.nnz << " non-zero elements." << std::endl;
    std::cout << "Generation time: " << elapsed.count() << " seconds." << std::endl;

    std::cout << "Generating right-hand side vector..." << std::endl;
    std::vector<double> b = generateRightHandSide(matrix_size);

    std::cout << "Saving matrix and right-hand side to files..." << std::endl;
    if (!saveCRSMatrixToFiles(A, matrix_file_prefix)) {
        std::cerr << "Failed to save matrix to files!" << std::endl;
        return 1;
    }

    if (!saveVectorToFile(b, rhs_file)) {
        std::cerr << "Failed to save right-hand side to file!" << std::endl;
        return 1;
    }

    std::cout << "\nStep 1.1: Memory usage analysis..." << std::endl;
    double full_memory = calculateMemoryUsage(A, true);
    double crs_memory = calculateMemoryUsage(A, false);
    double crs_percent = (crs_memory / full_memory) * 100.0;

    std::cout << "Memory requirements:\n"
        << "  - Full matrix format: " << full_memory << " MB\n"
        << "  - CRS format: " << crs_memory << " MB\n"
        << "  - CRS format uses " << crs_percent << "% of full format memory" << std::endl;

    std::ofstream memoryFile("memory_usage.txt", std::ios::app);
    if (memoryFile.is_open()) {
        memoryFile << "Matrix size: " << matrix_size << "x" << matrix_size
            << ", Diagonals: " << num_diagonals
            << ", NNZ: " << A.nnz << "\n";
        memoryFile << "Full format: " << full_memory << " MB, "
            << "CRS format: " << crs_memory << " MB, "
            << "CRS/Full: " << crs_percent << "%\n\n";
        memoryFile.close();
        std::cout << "Memory usage information saved to memory_usage.txt" << std::endl;
    }

    std::cout << "\nStep 2: Solving system using CPU conjugate gradient solver..." << std::endl;
    int cpu_iterations;
    double cpu_residual;
    std::vector<double> x_cpu = conjugateGradientCPU(A, b, tolerance, max_iterations,
        &cpu_iterations, &cpu_residual);

    std::cout << "\nStep 3: Solving system using GPU conjugate gradient solver..." << std::endl;

    if (!initGPU()) {
        std::cerr << "Failed to initialize GPU resources!" << std::endl;
        return 1;
    }

    int gpu_iterations;
    double gpu_residual;
    std::vector<double> x_gpu = conjugateGradientGPU(A, b, tolerance, max_iterations,
        &gpu_iterations, &gpu_residual);


    std::cout << "\nStep 4: Comparing CPU and GPU results..." << std::endl;

    std::cout << std::fixed << std::setprecision(12);
    std::cout << "CPU Solver:\n"
        << "  - Iterations: " << cpu_iterations << "\n"
        << "  - Final residual: " << cpu_residual << std::endl;

    std::cout << "GPU Solver:\n"
        << "  - Iterations: " << gpu_iterations << "\n"
        << "  - Final residual: " << gpu_residual << std::endl;

    double rel_diff = relativeError(x_cpu, x_gpu);
    std::cout << "Relative difference between CPU and GPU solutions: " << rel_diff << std::endl;


    std::cout << "\nVerifying CPU solution..." << std::endl;
    bool cpu_ok = verifySolution(A, x_cpu, b);
    std::cout << "CPU solution " << (cpu_ok ? "CORRECT" : "INCORRECT") << std::endl;

    std::cout << "Verifying GPU solution..." << std::endl;
    bool gpu_ok = verifySolution(A, x_gpu, b);
    std::cout << "GPU solution " << (gpu_ok ? "CORRECT" : "INCORRECT") << std::endl;


    std::cout << "\nPerformance comparison:" << std::endl;
    std::cout << "        | CPU       | GPU       | Speedup    " << std::endl;
    std::cout << "---------------------------------------------" << std::endl;

    double cpu_time_per_iter = 0.0; 
    double gpu_time_per_iter = 0.0; 

    std::cout << "Iterations | " << std::setw(9) << cpu_iterations << " | "
        << std::setw(9) << gpu_iterations << " | "
        << std::setw(10) << std::fixed << std::setprecision(2)
        << static_cast<double>(cpu_iterations) / gpu_iterations << "x" << std::endl;

    shutdownGPU();

    std::cout << "\nBenchmark completed!" << std::endl;

    return 0;
}