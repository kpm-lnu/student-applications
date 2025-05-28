//
// Created by anmode on 20.08.2024.
//

#ifndef FEM_ENGINE_MARCHING_CUBES_H
#define FEM_ENGINE_MARCHING_CUBES_H
#include "mesh_generator.h"
#include "sphere_scalar_field.h"

class MarchingCubes : public MeshGeneratorStrategy {
public:
    // Constructor that takes a reference to a scalar field
    explicit MarchingCubes(ScalarField& scalarField);

    // Generates and returns the vertices for the isosurface mesh
    std::vector<glm::vec3> generateMesh() override;

    [[nodiscard]] const std::vector<unsigned int>& getIndices() const override {
        static const std::vector<unsigned int> empty_vector;
        return empty_vector;
    };

    // Copy Constructor
    MarchingCubes(const MarchingCubes& other);

    // Copy Assignment Operator
    MarchingCubes& operator=(const MarchingCubes& other);

    // Move Constructor
    MarchingCubes(MarchingCubes&& other) noexcept;

    // Move Assignment Operator
    MarchingCubes& operator=(MarchingCubes&& other) noexcept;

    // set the scalar field
    void setScalarField(ScalarField& scalarField);

    // Destructor
    ~MarchingCubes() override = default;
private:
    ScalarField& scalarField; // Reference to the scalar field

    // Interpolates the vertex position between two points based on their scalar values
    glm::vec3 interpolateVertex(const glm::vec3& p1, const glm::vec3& p2,
                                               float valp1, float valp2);

    // Computes the cube index based on the scalar values at the cube corners
    int computeCubeIndex(const float values[8]) const;
};
#endif //FEM_ENGINE_MARCHING_CUBES_H
