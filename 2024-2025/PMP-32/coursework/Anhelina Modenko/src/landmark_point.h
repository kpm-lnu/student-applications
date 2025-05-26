//
// Created by anmode on 01.10.2024.
//

#ifndef FEM_ENGINE_LANDMARK_POINT_H
#define FEM_ENGINE_LANDMARK_POINT_H

#include <glm/glm.hpp>
#include <algorithm>
#include <vector>

class LandmarkPoint {
public:
    LandmarkPoint(float x = 0, float y = 0, float z = 0);

    glm::vec3 getPosition() const;
    void setPosition(float x, float y, float z);

private:
    glm::vec3 position;
};

class LandmarkPointNormalizer {
public:
    static void normalizePoints(std::vector<LandmarkPoint>& points) {
        if (points.empty()) return;

        float minX = std::numeric_limits<float>::max(), maxX = std::numeric_limits<float>::lowest();
        float minY = std::numeric_limits<float>::max(), maxY = std::numeric_limits<float>::lowest();
        float minZ = std::numeric_limits<float>::max(), maxZ = std::numeric_limits<float>::lowest();

        glm::vec3 centroid(0.0f);

        for (const auto& point : points) {
            const glm::vec3& pos = point.getPosition();
            minX = std::min(minX, pos.x);
            maxX = std::max(maxX, pos.x);
            minY = std::min(minY, pos.y);
            maxY = std::max(maxY, pos.y);
            minZ = std::min(minZ, pos.z);
            maxZ = std::max(maxZ, pos.z);
        }

        float verticalScaleFactor = 1.0f;

        // Determine the largest range to maintain aspect ratio
        float rangeX = maxX - minX;
        float rangeY = maxY - minY;
        float rangeZ = maxZ - minZ;
        float maxRange = std::max({rangeX, rangeY, rangeZ});

        for (auto& point : points) {
            glm::vec3 pos = point.getPosition();

            pos.x = ((pos.x - minX) / maxRange) * verticalScaleFactor;
            pos.y = (pos.y - minY) / maxRange;
            pos.z = (pos.z - minZ) / maxRange;

            point.setPosition(pos.x, pos.y, pos.z);
        }
    }

    static void flipPointsUpsideDown(std::vector<LandmarkPoint>& points) {
        for (auto& point : points) {
            glm::vec3 pos = point.getPosition();
            pos.y = -pos.y;
            point.setPosition(pos.x, pos.y, pos.z);
        }
    }
    
    static void scalePoints(std::vector<LandmarkPoint>& points, float scaleFactor) {
        for (auto& point : points) {
            glm::vec3 pos = point.getPosition();
            pos *= scaleFactor;
            point.setPosition(pos.x, pos.y, pos.z);
        }
    }
};




#endif //FEM_ENGINE_LANDMARK_POINT_H
