#ifndef DELTRIGEN_MESH_TRIANGULATION_DELAUNAY_BOWYERWATSON_HPP
#define DELTRIGEN_MESH_TRIANGULATION_DELAUNAY_BOWYERWATSON_HPP

#include <vector>
#include <unordered_set>
#include <unordered_map>

#include <boost/pool/pool_alloc.hpp>

#include <DelTriGen/Mesh/Triangulation/Tools/FWD.hpp>
#include <DelTriGen/Mesh/Triangulation/Configuration.hpp>
#include <DelTriGen/Mesh/Triangulation/Tools/ToTriangles.hpp>
#include <DelTriGen/Mesh/Triangulation/Tools/Triangulator.hpp>
#include <DelTriGen/Mesh/Triangulation/Tools/DelaunayCondition.hpp>
#include <DelTriGen/Mesh/Triangulation/Tools/WithSuperstructure.hpp>
#include <DelTriGen/Mesh/Triangulation/Tools/SuperstructureGenerator.hpp>

namespace DelTriGen::Mesh::Triangulation::Delaunay
{

using Tools::WithSuperstructureT;
using Tools::withSuperstructure;

using Tools::ToTrianglesT;
using Tools::toTriangles;

template <typename GeneratorT = Tools::SuperTriangle>
class BowyerWatson : public Tools::Triangulator<BowyerWatson<GeneratorT>>
{
public:

    using Point          = Primitives::Point;
    using Edge           = Primitives::Edge;
    using Triangle       = Primitives::Triangle;
    using Circle         = Primitives::Circle;
    using CoordinateType = Point::CoordinateType;
    using PointIndex     = Triangle::PointIndex;
    using Edges          = std::unordered_set<Edge
                                             , std::hash<Edge>
                                             , std::equal_to<Edge>
                                             , boost::fast_pool_allocator<Edge>>;

    using KeyValuePair   = std::pair<const Triangle, Circle>;
    using TrianglesMap   = std::unordered_map<Triangle, Circle
                                             , std::hash<Triangle>
                                             , std::equal_to<Triangle>
                                             , boost::fast_pool_allocator<KeyValuePair>>;

    using Points         = std::vector<Point, boost::pool_allocator<Point>>;
    using Triangles      = std::vector<Triangle, boost::pool_allocator<Triangle>>;
    using Polygon        = Edges;
    using Mesh           = std::pair<Points, TrianglesMap>;
    using IndexRange     = std::pair<PointIndex, PointIndex>;
    using SuperMesh      = std::pair<Mesh, const IndexRange>;

    template <Utilities::Concepts::InputRangeOf<Point> RangeT>
    static constexpr auto triangulate(RangeT&& points);

    template <Utilities::Concepts::InputRangeOf<Point> RangeT>
    static constexpr auto triangulate(RangeT&& points, WithSuperstructureT);
    
    static constexpr void refine(SuperMesh& superMesh, const Configuration& config);

    static constexpr void insert(Point point, SuperMesh& superMesh);

    template <Utilities::Concepts::InputRangeOf<Point> RangeT>
    static constexpr void insert(RangeT&& points, SuperMesh& superMesh);

    static constexpr void removeSuperstructure(SuperMesh& superMesh);

    static_assert(Concepts::Superstructure<Points, GeneratorT>);

private:

    static constexpr auto getSuperVerticePredicate(IndexRange idxRange) noexcept;
    static constexpr auto getSuperEdgePredicate(IndexRange idxRange) noexcept;
    static constexpr auto getSuperTriPredicate(IndexRange idxRange) noexcept;
    static constexpr auto getSuperKVPredicate(IndexRange idxRange) noexcept;

    static constexpr void insert(Point point, Mesh& mesh, ToTrianglesT
        , PointIndex shift = 0);

    template <Utilities::Concepts::InputRangeOf<Point> RangeT>
    static constexpr void insert(RangeT&& points, Mesh& mesh, ToTrianglesT
        , PointIndex shift = 0);

    template <std::predicate<const Triangle&> F>
    static constexpr void centroidRefinement(SuperMesh& superMesh, F condition);

    template <std::predicate<Edge> F>
    static constexpr void midpointRefinement(SuperMesh& superMesh, F condition);

    static constexpr void boundaryRefinement(SuperMesh& superMesh
        , const Configuration& config);
};



template <typename GeneratorT>
template <Utilities::Concepts::InputRangeOf<Primitives::Point> RangeT>
constexpr auto BowyerWatson<GeneratorT>::triangulate(RangeT&& points)
{
    auto&& superMesh = triangulate(points, withSuperstructure);
    auto& [mesh, idxRange] = superMesh;

    removeSuperstructure(superMesh);

    return mesh;
}

template <typename GeneratorT>
template <Utilities::Concepts::InputRangeOf<Primitives::Point> RangeT>
constexpr auto BowyerWatson<GeneratorT>::triangulate(RangeT&& points
    , WithSuperstructureT)
{
    using DelTriGen::Mesh::Tools::circumcircle;
    using Triangulation::Tools::SuperstructureGenerator;

    auto isFinite = [](Point point) noexcept
        {
            return std::isfinite(point[0]) && std::isfinite(point[1]);
        };

    Mesh mesh
    {
        Points(std::from_range, points | std::views::filter(isFinite)),
        TrianglesMap{}
    };

    auto& [vertices, trianglesMap] = mesh;
    auto almostEqual = [](Point lhs, Point rhs) noexcept
        {
            constexpr CoordinateType absTol = 1E-6;
            constexpr CoordinateType relTol = 1E-13;

            return lhs.almostEqual(rhs, absTol, relTol);
        };

    std::ranges::sort(vertices);
    auto&& [it, end] = std::ranges::unique(vertices, almostEqual);
    vertices.erase(it, end);
    auto uniqueSize = vertices.size();

    if (uniqueSize < 3)
    {
        return SuperMesh{ std::move(mesh), IndexRange{} };
    }

    auto&& superstructure = SuperstructureGenerator<GeneratorT>::generate(vertices);
    auto& [nodes, triangles] = superstructure;

    vertices.append_range(nodes);
    auto verticesSize = vertices.size();

    trianglesMap.max_load_factor(0.75f);
    trianglesMap.reserve(2 * verticesSize);
    
    for (auto& triangle : triangles)
    {
         triangle.shift(uniqueSize);
         trianglesMap.emplace(triangle, circumcircle(triangle, vertices));
    }

    auto unique = vertices | std::views::take(uniqueSize);
    insert(unique, mesh, toTriangles);

    return SuperMesh 
           {
               std::move(mesh),
               IndexRange{ uniqueSize, verticesSize }
           };
}

template<typename GeneratorT>
constexpr void BowyerWatson<GeneratorT>::refine(SuperMesh& superMesh
    , const Configuration& config)
{
    using DelTriGen::Mesh::Tools::area;
    using DelTriGen::Mesh::Tools::squaredLength;
    using DelTriGen::Mesh::Tools::gradient;

    auto& [mesh, idxRange]         = superMesh;
    auto& [vertices, trianglesMap] = mesh;

    switch (config.strategy)
    {
        case RefinementStrategy::Area:
        {
            auto isPartOfSuperStructure = getSuperTriPredicate(idxRange);
            auto condition =
                [maxArea = config.maxArea, &vertices, isPartOfSuperStructure]
                (const Triangle& tri) noexcept 
                {
                    return !isPartOfSuperStructure(tri) &&
                           area(tri, vertices) > maxArea;
                };

            centroidRefinement(superMesh, condition);

            break;
        }

        case RefinementStrategy::EdgeLength:
        {
            auto isPartOfSuperStructure = getSuperEdgePredicate(idxRange);
            auto condition =
                [maxLength = config.maxEdgeLength, &vertices, isPartOfSuperStructure]
                (Edge edge) noexcept
                {
                    return
                        !isPartOfSuperStructure(edge) &&
                        squaredLength(edge, vertices) > maxLength;
                };

            midpointRefinement(superMesh, condition);

            break;
        }

        case RefinementStrategy::Boundary:
        {
            boundaryRefinement(superMesh, config);

            break;
        }

        case RefinementStrategy::Gradient:
        {
            auto isPartOfSuperStructure = getSuperTriPredicate(idxRange);
            auto condition =
                [&config, &vertices, isPartOfSuperStructure]
                (const Triangle& tri) noexcept
                {
                    auto gradNorm2 = gradient(tri, config.f, vertices).squaredNorm();
                    auto coeff     = gradNorm2 / config.gradThreshold;

                    if (coeff > 1)
                    {
                        coeff = 1 + std::log2(coeff);
                    }
                    else
                    {
                        return false;
                    }

                    return
                        !isPartOfSuperStructure(tri) &&
                        coeff * area(tri, vertices) > config.areaThreshold;
                };

            centroidRefinement(superMesh, condition);

            break;
        }

        default:
            break;
    }

    if (config.removeSuperstructure)
    {
        removeSuperstructure(superMesh);
    }
}

template<typename GeneratorT>
constexpr void BowyerWatson<GeneratorT>::insert(Point point, SuperMesh& superMesh)
{
    insert(std::views::single(point), superMesh);
}

template<typename GeneratorT>
template<Utilities::Concepts::InputRangeOf<Primitives::Point> RangeT>
constexpr void BowyerWatson<GeneratorT>::insert(RangeT&& points, SuperMesh& superMesh)
{
    auto& [mesh, idxRange]         = superMesh;
    auto& [vertices, trianglesMap] = mesh;
    auto shift = vertices.size();

    vertices.append_range(points);
    insert(points, mesh, toTriangles, shift);
}

template<typename GeneratorT>
constexpr void BowyerWatson<GeneratorT>::removeSuperstructure(SuperMesh& superMesh)
{
    auto& [mesh, idxRange]         = superMesh;
    auto& [vertices, trianglesMap] = mesh;
    auto isPartOfSuperStructure    = getSuperKVPredicate(idxRange);

    std::erase_if(trianglesMap, isPartOfSuperStructure);
}

template<typename GeneratorT>
constexpr auto BowyerWatson<GeneratorT>::getSuperVerticePredicate(IndexRange idxRange)
    noexcept
{
    auto& [lb, ub] = idxRange;

    return [lb, ub](PointIndex index) noexcept
        {
            return index >= lb && index < ub;
        };
}

template<typename GeneratorT>
constexpr auto BowyerWatson<GeneratorT>::getSuperEdgePredicate(IndexRange idxRange)
    noexcept
{
    auto isPartOfSuperStructure = getSuperVerticePredicate(idxRange);

    return [isPartOfSuperStructure](Edge edge) noexcept
        {
            return isPartOfSuperStructure(edge[0]) ||
                   isPartOfSuperStructure(edge[1]);
        };
}

template<typename GeneratorT>
constexpr auto BowyerWatson<GeneratorT>::getSuperTriPredicate(IndexRange idxRange)
    noexcept
{
    auto isPartOfSuperStructure = getSuperVerticePredicate(idxRange);

    return [isPartOfSuperStructure](const Triangle& tri) noexcept
        {
            return isPartOfSuperStructure(tri[0]) ||
                   isPartOfSuperStructure(tri[1]) ||
                   isPartOfSuperStructure(tri[2]);
        };
}

template<typename GeneratorT>
constexpr auto BowyerWatson<GeneratorT>::getSuperKVPredicate(IndexRange idxRange)
    noexcept
{
    auto isPartOfSuperStructure = getSuperTriPredicate(idxRange);

    return [isPartOfSuperStructure](const KeyValuePair& pair) noexcept
        {
            auto& [triangle, circle] = pair;

            return isPartOfSuperStructure(triangle);
        };
}

template<typename GeneratorT>
constexpr void BowyerWatson<GeneratorT>::insert(Point point, Mesh& mesh
    , ToTrianglesT
    , PointIndex shift)
{
    insert(std::views::single(point), mesh, toTriangles, shift);
}

template<typename GeneratorT>
template<Utilities::Concepts::InputRangeOf<Primitives::Point> RangeT>
constexpr void BowyerWatson<GeneratorT>::insert(RangeT&& points, Mesh& mesh
    , ToTrianglesT
    , PointIndex shift)
{
    using DelTriGen::Mesh::Tools::circumcircle;
    using Tools::DelaunayCondition;

    auto& [vertices, trianglesMap] = mesh;

    Triangles badTriangles;
    Polygon polygon;

    for (auto point : points)
    {
        for (auto& [triangle, circle] : trianglesMap)
        {
            if (!DelaunayCondition::isAlmostSatisfied(circle, point))
            {
                badTriangles.emplace_back(triangle);

                for (auto& edge : triangle.edges())
                {
                    if (auto&& [it, inserted] = polygon.emplace(edge); !inserted)
                    {
                        polygon.erase(it);
                    }
                }
            }
        }

        for (auto& badTriangle : badTriangles)
        {
            trianglesMap.erase(badTriangle);
        }

        for (auto edge : polygon)
        {
            Triangle triangle(edge, shift);

            trianglesMap.emplace(triangle, circumcircle(triangle, vertices));
        }

        badTriangles.clear();
        polygon.clear();

        ++shift;        
    }
}

template<typename GeneratorT>
template<std::predicate<const Primitives::Triangle&> F>
constexpr void BowyerWatson<GeneratorT>::centroidRefinement(SuperMesh& superMesh
    , F condition)
{
    using DelTriGen::Mesh::Tools::centroid;

    auto& [mesh, idxRange]         = superMesh;
    auto& [vertices, trianglesMap] = mesh;

    Points centroids;

    do
    {
        if (centroids.size() > 0)
        {
            insert(centroids, superMesh);
            centroids.clear();
        }

        for (auto& [triangle, circle] : trianglesMap)
        {
            if (condition(triangle))
            {
                centroids.emplace_back(centroid(triangle, vertices));
            }
        }

    } while (centroids.size() > 0);
}

template<typename GeneratorT>
template<std::predicate<Primitives::Edge> F>
constexpr void BowyerWatson<GeneratorT>::midpointRefinement(SuperMesh& superMesh
    , F condition)
{
    using DelTriGen::Mesh::Tools::midpoint;

    auto& [mesh, idxRange]         = superMesh;
    auto& [vertices, trianglesMap] = mesh;

    auto toMidpoint = [&vertices](Edge edge) noexcept
        {
            return midpoint(edge, vertices);
        };

    Edges edges;

    do
    {
        if (edges.size() > 0)
        {
            insert(edges | std::views::transform(toMidpoint), superMesh);
            edges.clear();
        }

        for (auto& [triangle, circle] : trianglesMap)
        {
            for (auto& edge : triangle.edges())
            {
                if (condition(edge))
                {
                    edges.emplace(edge);
                }
            }
        }

    } while (edges.size() > 0);
}

template<typename GeneratorT>
constexpr void BowyerWatson<GeneratorT>::boundaryRefinement(SuperMesh& superMesh
    , const Configuration& config)
{
    using DelTriGen::Mesh::Tools::midpoint;
    using DelTriGen::Mesh::Tools::squaredLength;

    auto& [mesh, idxRange]         = superMesh;
    auto& [vertices, trianglesMap] = mesh;
    auto isPartOfSuperStructure    = getSuperTriPredicate(idxRange);
    auto isSuperEdge               = getSuperEdgePredicate(idxRange);
    auto maxLength                 = config.maxEdgeLength;

    Points midpoints;

    do
    {
        if (midpoints.size() > 0)
        {
            insert(midpoints, superMesh);
            midpoints.clear();
        }

        for (auto& [triangle, circle] : trianglesMap)
        {
            if (isPartOfSuperStructure(triangle))
            {
                for (auto& edge : triangle.edges())
                {
                    if (!isSuperEdge(edge) &&
                        squaredLength(edge, vertices) > maxLength)
                    {
                        midpoints.emplace_back(midpoint(edge, vertices));
                    }
                }
            }
        }

    } while (midpoints.size() > 0);
}

} // namespace DelTriGen::Mesh::Triangulation::Delaunay

#endif // DELTRIGEN_MESH_TRIANGULATION_DELAUNAY_BOWYERWATSON_HPP
