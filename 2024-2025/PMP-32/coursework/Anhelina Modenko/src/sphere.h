//
// Created by anmode on 10.08.2024.
//

#ifndef FEM_ENGINE_SPHERE_H
#define FEM_ENGINE_SPHERE_H

#include "shader.h"
#include <vector>

class Sphere {
public:
    Sphere(float radius, int sectorCount, int stackCount);
    Sphere(glm::vec3 position, float radius, Shader& shader);
    ~Sphere();
    Sphere& operator=(const Sphere& other);
    Sphere(const Sphere& other);
    Sphere(Sphere&& other) noexcept;
    Sphere& operator=(Sphere&& other) noexcept;

    void draw(const glm::mat4& view, const glm::mat4& projection);
    void setupMesh();
    void setRadius(float newRadius) {
        radius = newRadius;
        generateSphere();  // Re-generate vertices
        setupMesh();      // Update VBO with new data
    }
    void setModelMatrix(const glm::mat4& modelMatrix);
    glm::vec3 getPosition() const;
    void move(const glm::vec3& direction);

private:
    std::vector<GLfloat> vertices;
    GLuint VAO, VBO;
    float radius;
    glm::vec3 position;
    glm::mat4 model;
    int sectorCount;
    int stackCount;
    Shader& shader;

    void generateSphere();
};


#endif //FEM_ENGINE_SPHERE_H
