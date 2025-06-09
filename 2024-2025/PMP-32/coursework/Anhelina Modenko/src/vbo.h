#ifndef FEM_ENGINE_VBO_H
#define FEM_ENGINE_VBO_H
#define GL_SILENCE_DEPRECATION

#include <OpenGL/gl3.h>
#include <vector>

/**
 * @class VBO
 * @brief Represents a Vertex Buffer Object (VBO) in OpenGL.
 *
 * A VBO is used to store vertex data for rendering. It works with a VAO to specify how
 * the data is used in the rendering pipeline.
 */
class VBO {
public:
    GLuint val; ///< The VBO handle.
    GLenum usage; ///< Usage pattern (e.g., GL_STATIC_DRAW).

    /**
     * @brief Constructs a VBO with the specified usage pattern.
     * @param usage The usage pattern for the buffer (default: GL_STATIC_DRAW).
     */
    VBO(GLenum usage = GL_STATIC_DRAW);

    /**
     * @brief Generates the VBO.
     */
    void generate();

    /**
     * @brief Binds the VBO to the GL_ARRAY_BUFFER target.
     */
    void bind();

    /**
     * @brief Unbinds the VBO from the GL_ARRAY_BUFFER target.
     */
    void unbind();

    /**
     * @brief Uploads data to the VBO from a vector.
     * @tparam T Type of the data elements.
     * @param data The vector of data to upload.
     */
    template<typename T>
    void upload_data(const std::vector<T>& data) {
        bind();
        glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(T), data.data(), usage);
        unbind();
    }

    /**
     * @brief Uploads data to the VBO from a raw pointer and size.
     * @param data Pointer to the data.
     * @param size Size of the data in bytes.
     */
    template<typename T>
    void upload_data(const T* data, GLsizeiptr size) {
        bind();
        glBufferData(GL_ARRAY_BUFFER, size, data, usage);
        unbind();
    }

    /**
     * @brief Cleans up the VBO.
     */
    void cleanup();
};

#endif // FEM_ENGINE_VBO_H
