//
// Created by anmode on 11.08.2024.
//

#ifndef FEM_ENGINE_LINE_SEGMENT_BUFFER_H
#define FEM_ENGINE_LINE_SEGMENT_BUFFER_H

#include "shader.h"
#include "buffer_object.h"
#include <vector>
#include <glm/glm.hpp>

class LineSegmentBuffer {
public:
    ArrayObject VAO;
    BufferObject VBO; // Vertex Buffer Object
    Shader& shader;
    glm::mat4 model;
    size_t numLineSegments;
    std::vector<std::pair<glm::vec3, glm::vec3>> lineSegments;

    LineSegmentBuffer(const std::vector<std::pair<glm::vec3, glm::vec3>>& lineSegments, Shader& shader);
    void setupBuffer(const std::vector<std::pair<glm::vec3, glm::vec3>>& lineSegments);
    void draw(const glm::mat4& view, const glm::mat4& projection);
    ~LineSegmentBuffer();

    // Copy and move constructors/operators
    LineSegmentBuffer(const LineSegmentBuffer& other);
    LineSegmentBuffer& operator=(const LineSegmentBuffer& other);
    LineSegmentBuffer(LineSegmentBuffer&& other) noexcept;
    LineSegmentBuffer& operator=(LineSegmentBuffer&& other) noexcept;
};


#endif //FEM_ENGINE_LINE_SEGMENT_BUFFER_H
