#include <fstream>
#include <sstream>

#include <iostream>

#include "../Header files/Triangulation/Tools/PointsReader.h" 

int main()
{
	using namespace Triangulation::Structures;
	using namespace Triangulation::Tools;

	const std::string POINTS_FILE_PATH("Input/Points.edg");
	const std::string TRIANGULATION_FILE_PATH("Output/Triangulation.triang");

	std::ifstream poinstIfStream(POINTS_FILE_PATH);

	Points points = PointsReader::read(poinstIfStream);

	poinstIfStream.close();

	std::ostringstream plotTriangulation;

	plotTriangulation << R"(cd "Scripts/GNU Octave" && )"
		              << R"(octave -qfH --eval ")"
	                  << "plotTriangulation('../../" << POINTS_FILE_PATH
		              << "', '../../" << TRIANGULATION_FILE_PATH << "');\"";

	system(
		plotTriangulation.str()
		.c_str());
	
	return 0;
}
