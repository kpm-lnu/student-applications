#ifndef FEM_ENGINE_MESH_H
#define FEM_ENGINE_MESH_H

#include "shader.h"
#include "vao.h"
#include "vbo.h"
#include "ebo.h"
#include "mesh_generator.h"
#include <vector>

/**
 * @class Mesh
 * @brief Represents a 3D mesh and manages its VAO, VBO, and EBO for rendering.
 */
class Mesh {
public:
    /**
     * @brief Constructor.
     * @param shader A reference to the Shader object used for rendering.
     */
    explicit Mesh(Shader& shader);

    /**
     * @brief Destructor.
     */
    ~Mesh();

    /**
     * @brief Adds a triangle to the mesh.
     * @param v0 The first vertex.
     * @param v1 The second vertex.
     * @param v2 The third vertex.
     */
    void add_triangle(const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2);

    /**
     * @brief Draws the mesh.
     * @param view The view matrix.
     * @param projection The projection matrix.
     */
    void draw(const glm::mat4& view, const glm::mat4& projection);

    /**
     * @brief Resets the mesh by clearing vertices and indices.
     */
    void reset();

    /**
     * @brief Generates the mesh using the specified mesh generator.
     * @param meshGenerator A shared pointer to the MeshGeneratorStrategy.
     */
    void generate_mesh(const std::shared_ptr<MeshGeneratorStrategy>& meshGenerator);

    /**
    * @brief Sets a new shader for the mesh.
    * @param new_shader Reference to the new Shader object.
    */
    void set_shader(Shader& new_shader);

private:
    Shader& shader; ///< Reference to the Shader.
    VAO vao; ///< The Vertex Array Object.
    VBO vbo; ///< The Vertex Buffer Object.
    EBO ebo; ///< The Element Buffer Object.

    std::vector<glm::vec3> vertices; ///< List of vertices.
    std::vector<glm::vec3> normals; ///< Per-vertex normals.
    std::vector<unsigned int> indices; ///< List of indices.

    /**
     * @brief Sets up the VAO, VBO, and EBO.
     */
    void setup_mesh();

    /**
     * @brief Computes normals for the mesh.
     */
    void compute_normals();

    /**
     * @brief Cleans up allocated resources.
     */
    void cleanup();
};

#endif // FEM_ENGINE_MESH_H
