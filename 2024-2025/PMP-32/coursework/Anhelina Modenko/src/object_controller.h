//
// Created by anmode on 10.08.2024.
//

#ifndef FEM_ENGINE_OBJECT_CONTROLLER_H
#define FEM_ENGINE_OBJECT_CONTROLLER_H

#include <glm/glm.hpp>

class ObjectController {
public:
    ObjectController();
    void processInput();
    glm::vec3 getRotation();
    glm::vec3 getPosition();
    void setPosition(const glm::vec3& pos);
    void setRotation(const glm::vec3& rot);

private:
    glm::vec3 position;
    glm::vec3 rotation;

};


#endif //FEM_ENGINE_OBJECT_CONTROLLER_H
