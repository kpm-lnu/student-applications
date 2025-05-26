#include "matrix_generator.h"
#include <random>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>

CRSMatrix generateSymmetricDiagonalMatrix(int size, int num_diagonals) {
    num_diagonals = std::min(num_diagonals, size);

    CRSMatrix matrix;
    matrix.rows = size;
    matrix.cols = size;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(1.0, 2.0);

    int estimated_nnz = static_cast<int>(1.5 * size * num_diagonals);
    matrix.values.reserve(estimated_nnz);
    matrix.col_indices.reserve(estimated_nnz);
    matrix.row_ptr.resize(size + 1, 0);

    std::vector<int> diag_offsets(num_diagonals);
    diag_offsets[0] = 0;

    for (int i = 1; i < num_diagonals; ++i) {
        diag_offsets[i] = i;
    }

    std::sort(diag_offsets.begin(), diag_offsets.end());

    int nnz_count = 0;

    for (int row = 0; row < size; ++row) {
        matrix.row_ptr[row] = nnz_count;
        double row_sum = 0.0;

        for (int d = 1; d < num_diagonals; ++d) {
            int offset = diag_offsets[d];
            double value = dist(gen);

            int col = row + offset;
            if (col < size) {
                matrix.values.push_back(value);
                matrix.col_indices.push_back(col);
                nnz_count++;
                row_sum += std::abs(value);
            }

            col = row - offset;
            if (col >= 0) {
                matrix.values.push_back(value);
                matrix.col_indices.push_back(col);
                nnz_count++;
                row_sum += std::abs(value);
            }
        }

        double diag_value = 2.0 * row_sum + 10.0;
        matrix.values.push_back(diag_value);
        matrix.col_indices.push_back(row);
        nnz_count++;
    }

    matrix.row_ptr[size] = nnz_count;
    matrix.nnz = nnz_count;

    for (int row = 0; row < size; ++row) {
        int start = matrix.row_ptr[row];
        int end = matrix.row_ptr[row + 1];

        std::vector<std::pair<int, double>> row_data;
        for (int i = start; i < end; ++i) {
            row_data.emplace_back(matrix.col_indices[i], matrix.values[i]);
        }

        std::sort(row_data.begin(), row_data.end());

        for (int i = start; i < end; ++i) {
            matrix.col_indices[i] = row_data[i - start].first;
            matrix.values[i] = row_data[i - start].second;
        }
    }

    return matrix;
}


bool saveCRSMatrixToFiles(const CRSMatrix& matrix, const std::string& prefix) {
    std::ofstream valuesFile(prefix + "_values.txt");
    std::ofstream colIndicesFile(prefix + "_col_indices.txt");
    std::ofstream rowPtrFile(prefix + "_row_ptr.txt");

    if (!valuesFile || !colIndicesFile || !rowPtrFile) {
        std::cerr << "Failed to open output files!" << std::endl;
        return false;
    }

    valuesFile << matrix.rows << " " << matrix.cols << " " << matrix.nnz << std::endl;

    for (double val : matrix.values) {
        valuesFile << val << " ";
    }

    for (int idx : matrix.col_indices) {
        colIndicesFile << idx << " ";
    }

    for (int ptr : matrix.row_ptr) {
        rowPtrFile << ptr << " ";
    }

    return true;
}

CRSMatrix loadCRSMatrixFromFiles(const std::string& prefix) {
    CRSMatrix matrix;
    std::ifstream valuesFile(prefix + "_values.txt");
    std::ifstream colIndicesFile(prefix + "_col_indices.txt");
    std::ifstream rowPtrFile(prefix + "_row_ptr.txt");

    if (!valuesFile || !colIndicesFile || !rowPtrFile) {
        std::cerr << "Failed to open input files!" << std::endl;
        return matrix;
    }

    valuesFile >> matrix.rows >> matrix.cols >> matrix.nnz;

    matrix.values.resize(matrix.nnz);
    matrix.col_indices.resize(matrix.nnz);
    matrix.row_ptr.resize(matrix.rows + 1);

    for (int i = 0; i < matrix.nnz; ++i) {
        valuesFile >> matrix.values[i];
    }

    for (int i = 0; i < matrix.nnz; ++i) {
        colIndicesFile >> matrix.col_indices[i];
    }

    for (int i = 0; i <= matrix.rows; ++i) {
        rowPtrFile >> matrix.row_ptr[i];
    }

    return matrix;
}

std::vector<double> generateRightHandSide(int size) {
    std::vector<double> b(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(1.0, 10.0);

    for (int i = 0; i < size; ++i) {
        b[i] = dist(gen);
    }

    return b;
}

bool saveVectorToFile(const std::vector<double>& vec, const std::string& filename) {
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Failed to open output file: " << filename << std::endl;
        return false;
    }

    file << vec.size() << std::endl;
    for (double val : vec) {
        file << val << " ";
    }

    return true;
}

std::vector<double> loadVectorFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Failed to open input file: " << filename << std::endl;
        return std::vector<double>();
    }

    int size;
    file >> size;

    std::vector<double> vec(size);
    for (int i = 0; i < size; ++i) {
        file >> vec[i];
    }

    return vec;
}

double calculateMemoryUsage(const CRSMatrix& matrix, bool fullFormat) {
    double memoryUsage = 0.0;
    if (fullFormat) {
        memoryUsage = static_cast<double>(matrix.rows) * matrix.cols * sizeof(double);
    }
    else {
        memoryUsage = static_cast<double>(matrix.nnz) * sizeof(double)
            + matrix.nnz * sizeof(int)
            + (matrix.rows + 1) * sizeof(int);
    }

    return memoryUsage / (1024 * 1024);
}