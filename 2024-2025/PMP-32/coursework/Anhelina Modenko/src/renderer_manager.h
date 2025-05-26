#ifndef FEM_ENGINE_RENDERER_MANAGER_H
#define FEM_ENGINE_RENDERER_MANAGER_H

#include <vector>
#include <memory>
#include <tuple>
#include <OpenGL/gl3.h>
#include "vao.h"
#include "vbo.h"

/**
 * @class RendererManager
 * @brief Manages VAOs and VBOs, providing a simplified interface for OpenGL rendering operations.
 * Facade pattern to manage VAO and VBO
 */
class RendererManager {
public:
    /**
     * @brief Creates a VAO with a VBO and configures vertex attributes.
     * @param data Pointer to the vertex data.
     * @param data_size Size of the vertex data in bytes.
     * @param attributes A vector of attributes where each element defines
     *        (index, size, type, normalized, stride, offset).
     * @return A shared pointer to the created VAO.
     */
    std::shared_ptr<VAO> create_vao_with_vbo(const void* data, GLsizeiptr data_size,
                                             const std::vector<std::tuple<GLuint, GLint, GLenum, GLboolean, GLsizei, size_t>>& attributes);

    /**
     * @brief Draws the specified VAO.
     * @param vao The VAO to draw.
     * @param mode The drawing mode (e.g., GL_TRIANGLES).
     * @param vertex_count Number of vertices to draw.
     * @param index_type The type of the indices if using an EBO (default: GL_UNSIGNED_INT).
     * @param indices Pointer to the index data (default: nullptr).
     * @param instance_count Number of instances for instanced rendering (default: 1).
     */
    void draw_vao(const std::shared_ptr<VAO>& vao, GLenum mode, GLsizei vertex_count, GLenum index_type = GL_UNSIGNED_INT,
                  const void* indices = nullptr, GLuint instance_count = 1);

    /**
     * @brief Cleans up all managed VAOs and VBOs.
     */
    void cleanup();

private:
    std::vector<std::shared_ptr<VAO>> managed_vaos; ///< List of managed VAOs.
    std::vector<std::shared_ptr<VBO>> managed_vbos; ///< List of managed VBOs.
};

#endif // FEM_ENGINE_RENDERER_MANAGER_H
