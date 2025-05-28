//
// Created by anmode on 22.06.2024.
//

#ifndef FEM_ENGINE_SHADER_H
#define FEM_ENGINE_SHADER_H

#include <string>
#include <glm/glm.hpp>
#include <OpenGL/gl3.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <sstream>
#include <glm/gtc/type_ptr.hpp>

/*
 * Class to create and manage shader programs
 */
class Shader {
public:
    /*
     * Constructors
     */
    Shader();
    Shader(bool includeDefaultHeader,
           const char* vertexShaderPath,
           const char* fragShaderPath,
           const char* geoShaderPath = nullptr);
    ~Shader();

    /*
     * Methods to activate and cleanup the shader
     */
    void activate() const;
    void cleanup();

    /*
     * Static variables
     * defaultDirectory - directory where the default headers are stored
     * defaultHeaders - default headers to include in the shader
     */
    static std::string defaultDirectory;
    static std::stringstream defaultHeaders;

    /*
     * Get the id of the shader program
     */
    GLuint get_id() const;


    static std::string load_shader_source(const std::string& filePath, bool includeDefaultHeader = false);

    /*
     * Class to set uniform values in the shader
     *
     * Usage:
     * Shader::UniformSetter uniformSetter(shader.get_id());
     * uniformSetter.set_float("u_time", static_cast<float>(glfwGetTime()/1000000));
     *
     * Parameters:
     * @param programId - id of the shader program
     * @param name - name of the uniform variable
     * @param value - value to set
     * @param v1, v2, v3, v4 - values to set
     */
    class UniformSetter {
    public:
        UniformSetter(GLuint programId);

        void set_bool(const std::string& name, bool value) const;
        void set_int(const std::string& name, int value) const;
        void set_float(const std::string& name, float value) const;
        void set_2_float(const std::string& name, float v1, float v2) const;
        void set_2_float(const std::string& name, glm::vec2 value) const;
        void set_3_float(const std::string& name, float v1, float v2, float v3) const;
        void set_3_float(const std::string& name, glm::vec3 v) const;
        void set_4_float(const std::string& name, float v1, float v2, float v3, float v4) const;
        void set_4_float(const std::string& name, glm::vec4 v) const;
        //void set_4_float(const std::string& name, aiColor4D color)const;  ------- gonna have later for assimp model loading
        void set_3_mat(const std::string& name, glm::mat3 val) const;
        void set_4_mat(const std::string& name, glm::mat4 val )const;

    private:
        GLuint programId;
    };

    static GLuint compile_shader(bool includeDefaultHeader, const char* filePath, GLuint type);
    static void load_into_default(const char* filepath);
    static void clear_default();
    static char* load_shader_src(bool includeDefaultHeader, const char* filePath);

private:
    void generate(bool includeDefaultHeader,
                  const char* vertexShaderPath,
                  const char* fragShaderPath,
                  const char* geoShaderPath = nullptr);
    GLuint id;
    bool isLinked;

    friend class ShaderBuilder;
};

/*
 * Class to build shader programs
 */
class ShaderBuilder {
public:
    ShaderBuilder& set_vertex_shader(const std::string& filePath);
    ShaderBuilder& set_fragment_shader(const std::string& filePath);
    ShaderBuilder& set_geometry_shader(const std::string& filePath);

    Shader build(bool includeDefaultHeader = false);

private:
    std::string vertexShaderSource;
    std::string fragmentShaderSource;
    std::string geometryShaderSource;
};


#endif //FEM_ENGINE_SHADER_H
