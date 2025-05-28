#pragma once
#include <vector>
#include <string>

struct CRSMatrix {
    int rows;
    int cols;
    int nnz; 
    std::vector<double> values;
    std::vector<int> col_indices;
    std::vector<int> row_ptr;
};

CRSMatrix generateSymmetricDiagonalMatrix(int size, int num_diagonals);

bool saveCRSMatrixToFiles(const CRSMatrix& matrix, const std::string& prefix);

CRSMatrix loadCRSMatrixFromFiles(const std::string& prefix);

std::vector<double> generateRightHandSide(int size);

bool saveVectorToFile(const std::vector<double>& vec, const std::string& filename);

std::vector<double> loadVectorFromFile(const std::string& filename);

double calculateMemoryUsage(const CRSMatrix& matrix, bool fullFormat);