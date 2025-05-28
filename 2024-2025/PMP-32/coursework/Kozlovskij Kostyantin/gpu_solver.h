
#pragma once
#include "matrix_generator.h"
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>

std::vector<double> conjugateGradientGPU(
    const CRSMatrix& A,
    const std::vector<double>& b,
    double tolerance,
    int max_iterations,
    int* iterations_used = nullptr,
    double* final_residual = nullptr
);


bool initGPU();
void shutdownGPU();

void checkCudaError(cudaError_t result, const char* function, const char* file, int line);
void checkCuSparseError(cusparseStatus_t result, const char* function, const char* file, int line);
void checkCuBlasError(cublasStatus_t result, const char* function, const char* file, int line);


#define CHECK_CUDA(func) checkCudaError((func), #func, __FILE__, __LINE__)
#define CHECK_CUSPARSE(func) checkCuSparseError((func), #func, __FILE__, __LINE__)
#define CHECK_CUBLAS(func) checkCuBlasError((func), #func, __FILE__, __LINE__)