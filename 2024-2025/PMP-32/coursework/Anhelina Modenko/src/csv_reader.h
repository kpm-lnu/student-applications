//
// Created by anmode on 01.10.2024.
//

#ifndef FEM_ENGINE_CSV_READER_H
#define FEM_ENGINE_CSV_READER_H

#include <string>
#include <vector>
#include "landmark_point.h"

class CSVReader {
public:
    std::vector<LandmarkPoint> readCSV(const std::string& filename);
    std::vector<glm::vec3> glmReadCSV(const std::string& filename);
    static std::vector<glm::vec3> readBinary(const std::string& filename);
};

#endif //FEM_ENGINE_CSV_READER_H
