//
// Created by anmode on 28.06.2024.
//

#ifndef FEM_ENGINE_UBO_H
#define FEM_ENGINE_UBO_H
#define GL_SILENCE_DEPRECATION


#include "buffer_object.h"
#include "shader.h"
#include <GLFW/glfw3.h>

#include <vector>
#include <string>
#include "element.h"

/**
 * @class UBO;
 * @brief Represents a Uniform Buffer Object in OpenGL for structured data sharing between CPU and GPU.
 */
class UBO : public BufferObject {
public:
    static GLuint next_binding_pos; ///< Tracks the next binding position for UBOs.

    Element block; ///< Describes the structure and layout of the data stored in the UBO.
    unsigned int calculated_size; ///< Total size of the UBO in bytes.
    GLuint binding_pos; ///< Binding point for the UBO in the shader.

    /**
    * @brief Default constructor initializing an empty UBO.
    */
    UBO();

    /**
     * @brief Constructor initializing a UBO with a given set of elements.
     * @param elements A vector of shared pointers to Element objects.
     */
    UBO(const std::vector<std::shared_ptr<Element>>& elements);

    /**
    * @brief Links the UBO to a shader using the specified name.
    * @param shader The Shader to attach the UBO to.
    * @param name The name of the uniform block in the shader.
    */
    void attach_to_shader(const Shader& shader, const std::string& name);
    void add_element(const std::shared_ptr<Element>& element);
    void init_null_data(GLenum usage);
    void bind_range(GLuint offset = 0);
    unsigned int calc_size();
    void add_element(const Element& element);

    // Iteration variables
    GLuint offset;
    GLuint popped_offset;
    std::vector<std::pair<unsigned int, Element*>> index_stack; // stack to keep track of the nested indices
    int current_depth; // current size of the stack - 1

    void start_write();
    Element get_next_element();
    bool pop();

    template<typename T>
    void write_element(T* data);

    template<typename T>
    void write_array(T* first, unsigned int n);

    template<typename T, typename V>
    void write_array_container(T* container, unsigned int n);

    void advance_cursor(unsigned int n);
    void advance_array(unsigned int no_elements);
};

#endif //FEM_ENGINE_UBO_H
