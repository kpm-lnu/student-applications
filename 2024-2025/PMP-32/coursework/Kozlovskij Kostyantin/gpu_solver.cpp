#include "gpu_solver.h"
#include <iostream>
#include <chrono>
#include <cmath>

cusparseHandle_t cusparseHandle = nullptr;
cublasHandle_t cublasHandle = nullptr;

bool initGPU() {
    cusparseStatus_t cusparseStatus = cusparseCreate(&cusparseHandle);
    if (cusparseStatus != CUSPARSE_STATUS_SUCCESS) {
        std::cerr << "Failed to initialize cuSPARSE" << std::endl;
        return false;
    }

    cublasStatus_t cublasStatus = cublasCreate(&cublasHandle);
    if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
        cusparseDestroy(cusparseHandle);
        std::cerr << "Failed to initialize cuBLAS" << std::endl;
        return false;
    }

    return true;
}

void shutdownGPU() {
    if (cusparseHandle) {
        cusparseDestroy(cusparseHandle);
        cusparseHandle = nullptr;
    }

    if (cublasHandle) {
        cublasDestroy(cublasHandle);
        cublasHandle = nullptr;
    }
}

void checkCudaError(cudaError_t result, const char* function, const char* file, int line) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << " code=" << static_cast<unsigned int>(result)
            << " \"" << function << "\" " << cudaGetErrorString(result) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void checkCuSparseError(cusparseStatus_t result, const char* function, const char* file, int line) {
    if (result != CUSPARSE_STATUS_SUCCESS) {
        std::cerr << "cuSPARSE error at " << file << ":" << line << " code=" << static_cast<unsigned int>(result)
            << " \"" << function << "\"" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void checkCuBlasError(cublasStatus_t result, const char* function, const char* file, int line) {
    if (result != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS error at " << file << ":" << line << " code=" << static_cast<unsigned int>(result)
            << " \"" << function << "\"" << std::endl;
        exit(EXIT_FAILURE);
    }
}




std::vector<double> conjugateGradientGPU(
    const CRSMatrix& A,
    const std::vector<double>& b,
    double tolerance,
    int max_iterations,
    int* iterations_used,
    double* final_residual
) {
    auto start = std::chrono::high_resolution_clock::now();

    int n = A.rows;
    int nnz = A.nnz;

    double* d_val, * d_x, * d_b, * d_r, * d_p, * d_Ap;
    int* d_row, * d_col;

    CHECK_CUDA(cudaMalloc((void**)&d_val, nnz * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_row, (n + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_col, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_x, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_b, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_r, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_p, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_Ap, n * sizeof(double)));


    CHECK_CUDA(cudaMemcpy(d_val, A.values.data(), nnz * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_row, A.row_ptr.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_col, A.col_indices.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, b.data(), n * sizeof(double), cudaMemcpyHostToDevice));


    CHECK_CUDA(cudaMemset(d_x, 0, n * sizeof(double)));

    cusparseMatDescr_t descr;
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descr));
    CHECK_CUSPARSE(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));

    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecP, vecAp;
    void* dBuffer = nullptr;
    size_t bufferSize = 0;

    CHECK_CUSPARSE(cusparseCreateCsr(&matA, n, n, nnz,
        d_row, d_col, d_val,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));


    CHECK_CUDA(cudaMemcpy(d_r, d_b, n * sizeof(double), cudaMemcpyDeviceToDevice));

    CHECK_CUDA(cudaMemcpy(d_p, d_r, n * sizeof(double), cudaMemcpyDeviceToDevice));

    double rsold, rsnew, alpha, beta, residual;
    CHECK_CUBLAS(cublasDdot(cublasHandle, n, d_r, 1, d_r, 1, &rsold));

    double init_residual = std::sqrt(rsold);
    std::cout << "GPU Solver: Initial residual = " << init_residual << std::endl;

    CHECK_CUSPARSE(cusparseCreateDnVec(&vecP, n, d_p, CUDA_R_64F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecAp, n, d_Ap, CUDA_R_64F));

    double alpha_device = 1.0;
    double beta_device = 0.0;
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(
        cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha_device, matA, vecP, &beta_device, vecAp,
        CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));

    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    int iter;
    for (iter = 0; iter < max_iterations; ++iter) {
        CHECK_CUSPARSE(cusparseSpMV(
            cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha_device, matA, vecP, &beta_device, vecAp,
            CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));

        double dot_p_Ap;
        CHECK_CUBLAS(cublasDdot(cublasHandle, n, d_p, 1, d_Ap, 1, &dot_p_Ap));
        alpha = rsold / dot_p_Ap;

        CHECK_CUBLAS(cublasDaxpy(cublasHandle, n, &alpha, d_p, 1, d_x, 1));

        double neg_alpha = -alpha;
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, n, &neg_alpha, d_Ap, 1, d_r, 1));

        CHECK_CUBLAS(cublasDdot(cublasHandle, n, d_r, 1, d_r, 1, &rsnew));

        residual = std::sqrt(rsnew);
        if (residual < tolerance) {
            if (iterations_used) *iterations_used = iter + 1;
            if (final_residual) *final_residual = residual;
            break;
        }

        beta = rsnew / rsold;
        CHECK_CUBLAS(cublasDscal(cublasHandle, n, &beta, d_p, 1));
        double one = 1.0;
        CHECK_CUBLAS(cublasDaxpy(cublasHandle, n, &one, d_r, 1, d_p, 1));

        rsold = rsnew;

        if ((iter + 1) % 100 == 0) {
            std::cout << "GPU Solver: Iteration " << (iter + 1)
                << ", Residual = " << residual << std::endl;
        }
    }

    std::vector<double> x(n);
    CHECK_CUDA(cudaMemcpy(x.data(), d_x, n * sizeof(double), cudaMemcpyDeviceToHost));

    CHECK_CUSPARSE(cusparseDestroyDnVec(vecP));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecAp));
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroyMatDescr(descr));
    CHECK_CUDA(cudaFree(dBuffer));
    CHECK_CUDA(cudaFree(d_val));
    CHECK_CUDA(cudaFree(d_row));
    CHECK_CUDA(cudaFree(d_col));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_r));
    CHECK_CUDA(cudaFree(d_p));
    CHECK_CUDA(cudaFree(d_Ap));

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    if (iter == max_iterations) {
        std::cout << "GPU Solver: Maximum iterations reached without convergence." << std::endl;
        if (iterations_used) *iterations_used = max_iterations;
        if (final_residual) *final_residual = residual;
    }
    else {
        std::cout << "GPU Solver: Converged in " << (iter + 1) << " iterations." << std::endl;
    }

    std::cout << "GPU Solver: Elapsed time = " << elapsed.count() << " seconds." << std::endl;

    return x;
}