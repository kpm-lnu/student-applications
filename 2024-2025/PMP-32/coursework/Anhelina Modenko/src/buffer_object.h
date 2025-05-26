//
// Created by anmode on 22.06.2024.
//

#ifndef FEM_ENGINE_BUFFER_OBJECT_H
#define FEM_ENGINE_BUFFER_OBJECT_H


#include <OpenGL/gl3.h>
#include <GLFW/glfw3.h>
#include <map>

/**
* @brief Links the UBO to a shader using the specified name.
* @param shader The Shader to attach the UBO to.
* @param name The name of the uniform block in the shader.
*/
class BufferObject {
public:
    GLuint val; ///< The OpenGL ID for the buffer object.
    GLenum type; ///< The type of the buffer (e.g., GL_ARRAY_BUFFER, GL_ELEMENT_ARRAY_BUFFER).

    /**
     * @brief Constructor initializing the buffer type.
     * @param type The OpenGL buffer type.
     */
    BufferObject(GLenum type = GL_ARRAY_BUFFER)
            : type(type) {}

    /**
     * @brief Generates the OpenGL buffer.
     */
    void generate() {
        glGenBuffers(1, &val);
    }

    /**
     * @brief Binds the buffer for usage.
     */
    void bind() {
        glBindBuffer(type, val);
    }

    // set data (glBufferData)
    template<typename T>
    void setData(GLuint noElements, T* data, GLenum usage) {
        glBufferData(type, noElements, data, usage);
    }

    // update data (glBufferSubData)
    template<typename T>
    void updateData(GLintptr offset, GLuint noElements, T* data) {
        glBufferSubData(type, offset, noElements * sizeof(T), data);
    }

    // set attribute pointers
    template<typename T>
    void setAttPointer(GLuint idx, GLint size, GLenum type, GLuint stride, GLuint offset, GLuint divisor = 0) {
        glVertexAttribPointer(idx, size, type, GL_FALSE, stride * sizeof(T), (void*)(offset * sizeof(T)));
        glEnableVertexAttribArray(idx);
        if (divisor > 0) {
            // reset _idx_ attribute every _divisor_ iteration (instancing)
            glVertexAttribDivisor(idx, divisor);
        }
    }

    // clear buffer objects (bind 0)
    void clear() {
        glBindBuffer(type, 0);
    }

    // cleanup
    void cleanup() {
        glDeleteBuffers(1, &val);
    }
};

/*
    class for array objects
    - VAO
*/
class ArrayObject {
public:
    // value/location
    GLuint val;

    // map of names to buffers
    std::map<const char*, BufferObject> buffers;

    // get buffer (override [])
    BufferObject& operator[](const char* key) {
        return buffers[key];
    }

    // generate object
    void generate() {
        glGenVertexArrays(1, &val);
    }

    // bind
    void bind() {
        glBindVertexArray(val);
    }

    // draw arrays
    void draw(GLenum mode, GLuint first, GLuint count, GLuint instancecount = 1) {
        glDrawArraysInstanced(mode, first, count, instancecount);
    }

    // draw
    void draw(GLenum mode, GLuint count, GLenum type, GLint indices, GLuint instancecount = 1) {
        glDrawElementsInstanced(mode, count, type, (void*)indices, instancecount);
    }

    // cleanup
    void cleanup() {
        glDeleteVertexArrays(1, &val);
        for (auto& pair : buffers) {
            pair.second.cleanup();
        }
    }

    // clear array object (bind 0)
    static void clear() {
        glBindVertexArray(0);
    }
};


#endif //FEM_ENGINE_BUFFER_OBJECT_H
