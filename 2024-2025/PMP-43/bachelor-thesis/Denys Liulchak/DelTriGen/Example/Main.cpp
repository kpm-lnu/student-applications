#include <iostream>
#include <fstream>
#include <sstream>
#include <string_view>

#include <DelTriGen/Mesh/Triangulation/Tools/SuperTriangle.hpp>
#include <DelTriGen/Mesh/Triangulation/Tools/SuperRectangle.hpp>
#include <DelTriGen/Mesh/Triangulation/Delaunay/BowyerWatson.hpp>
#include <DelTriGen/Mesh/Triangulation/Tools/ConfigurationBuilder.hpp>

#include "IO/PointsReader.hpp"
#include "IO/RangeWriter.hpp"

int main()
{
    using namespace DelTriGen::Mesh::Tools;
    using namespace DelTriGen::Mesh::Primitives;
    using namespace DelTriGen::Mesh::Triangulation;
    using namespace DelTriGen::Mesh::Triangulation::Tools;
    using namespace DelTriGen::Mesh::Triangulation::Delaunay;

    using namespace Example::IO;

    using CoordinateType = Point::CoordinateType;
    using Triangulator   = BowyerWatson<SuperRectangle>;

    std::string_view pointsIfPath("./Example/Input/Points.txt");
    std::ifstream pointsIfStream(pointsIfPath.data());
    
    std::vector points(std::from_range, PointsReader::makeView(pointsIfStream));

    if (!pointsIfStream &&
        !pointsIfStream.eof())
    {
        std::cerr << "An error occurred while reading data from file \"" 
                  << pointsIfPath << "\"\n";

        return 0;
    }

    std::string_view verticesOfPath("Example/Output/Vertices.txt");
    std::string_view trianglesOfPath("Example/Output/Triangles.txt");
    std::ofstream verticesOfStream;
    std::ofstream trianglesOfStream;
    std::ostringstream plotMesh;

    plotMesh << "octave -qfH -p \"Scripts/Octave\" -e \""
             << "plot_tri('" << verticesOfPath << "', '" << trianglesOfPath << "');\""
             << '\0';

    ConfigurationBuilder builder;
    bool removedSuperstructure   = false;
    RefinementStrategy strategy  = RefinementStrategy::None;
    CoordinateType maxArea       = 0;
    CoordinateType maxEdgeLength = 0;
    CoordinateType gradThreshold = 0;
    CoordinateType areaThreshold = 0;
    auto gradF = [](Point point) noexcept
        {
            auto&& [x, y] = point.coordinates();

            return (x + y) * (x * x + y * y - 1);
        };

    auto&& superMesh =
        Triangulator::triangulate(points, withSuperstructure);

    auto& [mesh, idxRange]         = superMesh;
    auto& [vertices, trianglesMap] = mesh;

    auto findMinAngle = [&vertices](const Triangle& triangle)
        {
            return std::ranges::min(angles(triangle, vertices))
                * 180 / std::numbers::pi_v<CoordinateType>;
        };

    auto findArea = [&vertices](const Triangle& triangle)
        {
            return area(triangle, vertices);
        };

    auto triangles = trianglesMap | std::views::keys;
    auto minAngles = triangles | std::views::transform(findMinAngle);
    auto triAreas  = triangles | std::views::transform(findArea);

    int choice = 0;
    
    while (true)
    {
        std::cout << "\n=== Menu ===\n"
                  << "1. Toggle remove superstructure (currently "
                      << (removedSuperstructure ? "ON" : "OFF") << ")\n"
                  << "2. Set refinement strategy\n"
                  << "3. Set max area threshold\n"
                  << "4. Set max edge length threshold\n"
                  << "5. Set gradient threshold\n"
                  << "6. Set area threshold for gradient refinement\n"
                  << "7. Run refinement step\n"
                  << "8. Exit\n"
                  << "Enter choice: ";

        if (!(std::cin >> choice))
        {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << "Invalid input, please enter a number.\n";

            continue;
        }

        switch (choice)
        {
            case 1:
            {
                removedSuperstructure = !removedSuperstructure;
                std::cout << "removeSuperstructure = " 
                          << (removedSuperstructure ? "true" : "false") << "\n";

                break;
            }

            case 2:
            {
                int strat = 0;

                std::cout << "Select strategy:\n"
                          << "0: None\n"
                          << "1: Area\n"
                          << "2: EdgeLength\n"
                          << "3: Boundary\n"
                          << "4: Gradient\n"
                          << "Enter: ";

                if (std::cin >> strat && strat >= 0 && strat <= 4)
                {
                    strategy = static_cast<RefinementStrategy>(strat);
                }
                else
                {
                    std::cout << "Invalid strategy index\n";
                    std::cin.clear();
                }
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

                break;
            }

            case 3:
            {
                std::cout << "Enter maxArea: ";
                std::cin >> maxArea;

                break;
            }
            
            case 4:
            {
                std::cout << "Enter maxEdgeLength: ";
                std::cin >> maxEdgeLength;

                break;
            }

            case 5:
            {
                std::cout << "Enter gradThreshold: ";
                std::cin >> gradThreshold;

                break;
            }

            case 6:
            {
                std::cout << "Enter areaThreshold: ";
                std::cin >> areaThreshold;

                break;
            }

            case 7:
            {
                verticesOfStream.open(verticesOfPath.data());
                trianglesOfStream.open(trianglesOfPath.data());

                builder.removeSuperstructure(removedSuperstructure)
                    .strategy(strategy)
                    .maxArea(maxArea)
                    .maxEdgeLength(maxEdgeLength)
                    .gradThreshold(gradThreshold)
                    .areaThreshold(areaThreshold)
                    .f(gradF);
    
                Triangulator::refine(superMesh, builder.build());

                auto triCount = triangles.size();
                auto minAngle = std::ranges::min(minAngles);
                auto avgAngle = std::ranges::fold_left(minAngles, 0, std::plus{}) / triCount;
                auto minArea  = std::ranges::min(triAreas);
                auto avgArea  = std::ranges::fold_left(triAreas, 0, std::plus{}) / triCount;

                std::cout << "trianlges count: " << triCount << '\n'
                          << "min angle:       " << minAngle << '\n'
                          << "avg angle:       " << avgAngle << '\n'
                          << "min area:        " << minArea  << '\n'
                          << "avg area:        " << avgArea  << '\n';

                RangeWriter::write(verticesOfStream, vertices);
                RangeWriter::write(trianglesOfStream, triangles);
        
                if (!verticesOfStream ||
                    !trianglesOfStream)
                {
                    std::cerr << "An error occurred while writing to the file \n";

                    return 0;
                }

                verticesOfStream.close();
                trianglesOfStream.close();

                std::system(plotMesh.view().data());

                break;
            }

            case 8:
            {
                std::cout << "Exiting.\n";

                return 0;
            }

            default:
            {
                std::cout << "Unknown choice, please try again.\n";
            }
        }
    }
}
