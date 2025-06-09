#include "../../../Header files/Triangulation/Tools/DelaunayCondition.h"

namespace Triangulation::Tools
{
	bool DelaunayCondition::isSatisfied(const Edge& edge, const Point& point0, const Point& point2)
	{
		double x1 = edge.point1->x;
		double y1 = edge.point1->y;

		double x3 = edge.point2->x;
		double y3 = edge.point2->y;

		double deltaX0X1 = point0.x - x1;
		double deltaX0X3 = point0.x - x3;
		double deltaY0Y1 = point0.y - y1;
		double deltaY0Y3 = point0.y - y3;

		double sAlpha = deltaX0X1 * deltaX0X3 + deltaY0Y1 * deltaY0Y3;

		double deltaX2X1 = point2.x - x1;
		double deltaX2X3 = point2.x - x3;
		double deltaY2Y1 = point2.y - y1;
		double deltaY2Y3 = point2.y - y3;

		double sBeta = deltaX2X1 * deltaX2X3 + deltaY2Y1 * deltaY2Y3;

		if (sAlpha < 0 && sBeta < 0)
			return false;

		if (sAlpha >= 0 && sBeta >= 0)
			return true;

		double controlValue = (deltaX0X1 * deltaY0Y3 - deltaX0X3 * deltaY0Y1) * sBeta +
			                  (deltaX2X3 * deltaY2Y1 - deltaX2X1 * deltaY2Y3) * sAlpha;

		return controlValue >= 0;
	}
}
