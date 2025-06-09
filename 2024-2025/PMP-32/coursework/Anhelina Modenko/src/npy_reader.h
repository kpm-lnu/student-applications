#ifndef FEM_ENGINE_NPY_READER_H
#define FEM_ENGINE_NPY_READER_H

#include <vector>
#include <string>
#include <map>
#include <glm/glm.hpp>

struct FaceMetadata {
    std::map<std::string, std::vector<int>> parts;
    std::string model;
    int pointCount;
    std::vector<int> frameIndices;
    int totalVideoFrames = 0;
    int frameSkip = 1;
};

struct FrameLandmarkData {
    std::vector<glm::vec3> landmarks;
};

struct LandmarkSequence {
    std::vector<FrameLandmarkData> frames;
    FaceMetadata metadata;
};

class NpyReader {
public:
    static LandmarkSequence readLandmarkNpy(const std::string& filepath);
    static std::vector<glm::vec3> readSingleFrameNpy(const std::string& filename);
};


#endif