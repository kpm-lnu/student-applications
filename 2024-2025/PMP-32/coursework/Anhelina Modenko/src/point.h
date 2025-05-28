//
// Created by anmode on 22.06.2024.
//

#ifndef FEM_ENGINE_POINT_H
#define FEM_ENGINE_POINT_H

struct Point {
    int x;
    int y;
    int z;
    Point(int x, int y, int z) : x(x), y(y), z(z) {}
    Point() : x(0), y(0), z(0) {}
};

#endif //FEM_ENGINE_POINT_H
