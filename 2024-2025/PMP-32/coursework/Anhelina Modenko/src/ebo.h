#ifndef FEM_ENGINE_EBO_H
#define FEM_ENGINE_EBO_H

#include <OpenGL/gl3.h>
#include <vector>

/**
 * @class EBO
 * @brief Manages Element Buffer Objects (EBO) in OpenGL, which store indices
 *        for indexed rendering.
 *
 * The EBO class encapsulates the functionality for creating, binding,
 * uploading, and cleaning up index data in OpenGL. It is designed to work
 * seamlessly with VAOs and VBOs for efficient rendering of 3D meshes.
 */
class EBO {
public:
    /**
     * @brief Constructor for EBO.
     *        Initializes the EBO ID to zero.
     */
    EBO();

    /**
     * @brief Destructor for EBO.
     *        Cleans up GPU resources if allocated.
     */
    ~EBO();

    /**
     * @brief Generates the EBO by creating a buffer on the GPU.
     *
     * This must be called before using other methods like bind or upload_data.
     */
    void generate();

    /**
     * @brief Binds the EBO to the GL_ELEMENT_ARRAY_BUFFER target.
     *
     * Ensures that the EBO is ready for use in OpenGL operations.
     */
    void bind() const;

    /**
     * @brief Unbinds any currently bound EBO.
     *
     * Useful to ensure no EBO is accidentally modified by subsequent operations.
     */
    static void unbind();

    /**
     * @brief Uploads index data to the EBO.
     * @param indices A vector of unsigned integers representing the indices.
     * @param usage The usage pattern for the data (e.g., GL_STATIC_DRAW).
     *
     * Transfers the index data from the CPU to the GPU for use in rendering.
     * The usage parameter specifies the intended usage pattern of the data
     * for optimization purposes.
     */
    void upload_data(const std::vector<unsigned int>& indices, GLenum usage = GL_STATIC_DRAW);

    /**
     * @brief Cleans up the EBO by deleting its GPU buffer.
     *
     * Should be called when the EBO is no longer needed to free GPU resources.
     */
    void cleanup();
    GLuint id; ///< OpenGL ID of the EBO.

private:
};

#endif // FEM_ENGINE_EBO_H
