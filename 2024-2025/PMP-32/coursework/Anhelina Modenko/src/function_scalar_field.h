//
// Created by anmode on 26.08.2024.
//

#ifndef FEM_ENGINE_FUNCTION_SCALAR_FIELD_H
#define FEM_ENGINE_FUNCTION_SCALAR_FIELD_H


#include "sphere_scalar_field.h"
#include <vector>
#include <glm/glm.hpp>

class FunctionScalarField : public ScalarField {
public:
    FunctionScalarField(int width, int height, int depth, float isoLevel);

    void updateField(float time);
    void setSourcePoints(const std::vector<glm::vec3>& points);

private:
    std::vector<glm::vec3> sourcePoints;
    float fTime;

    float fSample1(float x, float y, float z) const;
    void updateSourcePoints();
};


#endif //FEM_ENGINE_FUNCTION_SCALAR_FIELD_H
