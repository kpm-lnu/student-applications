//
// Created by anmode on 12.11.2024.
//

#ifndef FEM_ENGINE_SPHERE_BUFFER_H
#define FEM_ENGINE_SPHERE_BUFFER_H

#include "shader.h"
#include <vector>
#include <OpenGL/gl3.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

class SphereBuffer {
public:
    SphereBuffer(const std::vector<glm::vec3>& positions, Shader& shader, float scale = 1.0f, bool invertY = false);
    ~SphereBuffer();
    SphereBuffer(const SphereBuffer& other);
    SphereBuffer& operator=(const SphereBuffer& other);
    SphereBuffer(SphereBuffer&& other) noexcept;
    SphereBuffer& operator=(SphereBuffer&& other) noexcept;

    void draw(const glm::mat4& view, const glm::mat4& projection);
    void setupBuffer();
    void setScale(float newScale);
    void updatePositions(const std::vector<glm::vec3>& newPositions);

private:
    std::vector<glm::vec3> instancePositions;
    float scale;
    bool invertY;
    std::vector<float> vertices;

    GLuint VAO, VBO, instanceVBO;
    Shader& shader;

    void cleanup();
};


#endif //FEM_ENGINE_SPHERE_BUFFER_H
