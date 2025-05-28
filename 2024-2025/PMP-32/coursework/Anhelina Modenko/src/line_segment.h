//
// Created by anmode on 10.08.2024.
//

#ifndef FEM_ENGINE_LINE_SEGMENT_H
#define FEM_ENGINE_LINE_SEGMENT_H

#include "shader.h"
#include <vector>

class LineSegment {
public:
    std::vector<glm::vec3> points;  // Store the positions of the sphere centers
    unsigned int VAO, VBO;
    Shader& shader;
    glm::mat4 model;

    LineSegment(const std::vector<glm::vec3>& points, Shader& shader);
    void setupMesh(const std::vector<glm::vec3>& points);
    void draw(const glm::mat4& view, const glm::mat4& projection);
    ~LineSegment();

    // Copy and move constructors/operators
    LineSegment(const LineSegment& other);
    LineSegment& operator=(const LineSegment& other);
    LineSegment(LineSegment&& other) noexcept;
    LineSegment& operator=(LineSegment&& other) noexcept;
};


#endif //FEM_ENGINE_LINE_SEGMENT_H
