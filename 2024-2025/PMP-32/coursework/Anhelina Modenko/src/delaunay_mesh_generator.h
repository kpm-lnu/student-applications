#ifndef FEM_ENGINE_DELAUNAY_MESH_GENERATOR_H
#define FEM_ENGINE_DELAUNAY_MESH_GENERATOR_H

#include <vector>
#include <glm/glm.hpp>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Alpha_shape_vertex_base_3.h>
#include <CGAL/Alpha_shape_cell_base_3.h>
#include <CGAL/Triangulation_data_structure_3.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Alpha_shape_3.h>

#include "mesh_generator.h"

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef K::Point_3 Point;

typedef CGAL::Alpha_shape_vertex_base_3<K> Vb;
typedef CGAL::Alpha_shape_cell_base_3<K> Fb;
typedef CGAL::Triangulation_data_structure_3<Vb, Fb> Tds;
typedef CGAL::Delaunay_triangulation_3<K, Tds> Delaunay3;
typedef CGAL::Alpha_shape_3<Delaunay3> Alpha_shape_3;

typedef Alpha_shape_3::Vertex_handle Vertex_handle;
typedef Alpha_shape_3::Facet Facet;

class DelaunayMeshGenerator : public MeshGeneratorStrategy {
public:
    explicit DelaunayMeshGenerator(const std::vector<glm::vec3>& points);
    std::vector<glm::vec3> generateMesh() override;
    const std::vector<unsigned int>& getIndices() const override;

private:
    std::vector<glm::vec3> inputPoints;
    std::vector<glm::vec3> vertices;
    std::vector<unsigned int> indices;

    void removeDuplicatePoints();
};

#endif