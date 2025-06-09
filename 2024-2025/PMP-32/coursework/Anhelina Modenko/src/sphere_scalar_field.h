//
// Created by anmode on 11.08.2024.
//

#ifndef FEM_ENGINE_SPHERE_SCALAR_FIELD_H
#define FEM_ENGINE_SPHERE_SCALAR_FIELD_H

#include <vector>
#include <glm/glm.hpp>
//#include "scalar_field.h"

class ScalarField {
public:
    ScalarField(int width, int height, int depth, float isoLevel);
    void setPointValue(int x, int y, int z, float value);
    float getPointValue(int x, int y, int z) const;
    void generateFieldFromSpheres(const std::vector<glm::vec3>& sphereCenters, float sphereRadius);
    void updateFieldForSphere(const glm::vec3& sphereCenter, float sphereRadius, const glm::vec3& prevCenter);

    int getWidth() const { return width; }
    int getHeight() const { return height; }
    int getDepth() const { return depth; }
    float getIsoLevel() const { return isoLevel; }

private:
protected:
    int width, height, depth;
    float isoLevel;
    std::vector<float> data;
};
#endif //FEM_ENGINE_SPHERE_SCALAR_FIELD_H
