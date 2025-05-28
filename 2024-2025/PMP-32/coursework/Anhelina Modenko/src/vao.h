#ifndef FEM_ENGINE_VAO_H
#define FEM_ENGINE_VAO_H
#define GL_SILENCE_DEPRECATION

#include <OpenGL/gl3.h>
#include <vector>
#include <map>
#include <memory>
#include "vbo.h"

/**
 * @class VAO
 * @brief Represents a Vertex Array Object (VAO) in OpenGL.
 *
 * A VAO encapsulates all the state needed to supply vertex data to OpenGL.
 * It references Vertex Buffer Objects (VBOs) and their associated vertex attribute pointers.
 *
 * @relation uses VBO for storing vertex data.
 */
class VAO {
public:
    GLuint val; ///< The VAO handle.

    /**
     * @brief Constructs and initializes the VAO.
     */
    VAO();

    /**
     * @brief Generates the VAO.
     */
    void generate();

    /**
     * @brief Binds the VAO.
     */
    void bind();

    /**
     * @brief Unbinds the VAO.
     */
    void unbind();

    /**
     * @brief Adds a VBO and sets its attributes.
     * @param vbo The VBO containing vertex data.
     * @param attributes A vector of attributes where each element defines
     *        (index, size, type, normalized, stride, offset).
     */
    void add_vbo(const std::shared_ptr<VBO>& vbo,
                 const std::vector<std::tuple<GLuint, GLint, GLenum, GLboolean, GLsizei, size_t>>& attributes);

    /**
     * @brief Draws the VAO.
     * @param mode The primitive drawing mode (e.g., GL_TRIANGLES).
     * @param count The number of vertices to draw.
     * @param type The type of the indices if using an EBO.
     * @param indices Pointer to the index data if using an EBO.
     * @param instance_count Number of instances for instanced rendering.
     */
    void draw(GLenum mode, GLsizei count, GLenum type = GL_UNSIGNED_INT, const void* indices = nullptr, GLuint instance_count = 1);

    /**
     * @brief Cleans up the VAO and its associated VBOs.
     */
    void cleanup();

private:
    std::map<GLuint, std::shared_ptr<VBO>> vbos; ///< Map of attribute indices to VBOs.
};

#endif // FEM_ENGINE_VAO_H
