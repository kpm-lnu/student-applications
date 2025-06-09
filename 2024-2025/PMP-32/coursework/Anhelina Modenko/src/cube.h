//
// Created by anmode on 10.08.2024.
//

#ifndef FEM_ENGINE_CUBE_H
#define FEM_ENGINE_CUBE_H

#include <OpenGL/gl3.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "shader.h"

class Cube {
public:
    Cube(const glm::vec3& position, float size, Shader& shader);
    ~Cube();

    Cube(const Cube& other);
    Cube& operator=(const Cube& other);

    Cube(Cube&& other) noexcept;
    Cube& operator=(Cube&& other) noexcept;

    void setPosition(const glm::vec3& position);
    void setModelMatrix(const glm::mat4& modelMatrix);
    void setRotation(float angle, const glm::vec3& axis);
    void draw(const glm::mat4& view, const glm::mat4& projection);

    glm::vec3 position;
private:
    void setupMesh();
    void cleanup();

    float size;
    glm::mat4 model;
    Shader& shader;

    GLuint VAO, VBO, EBO;
};

#endif //FEM_ENGINE_CUBE_H
