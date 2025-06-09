//
// Created by anmode on 26.08.2024.
//

#ifndef FEM_ENGINE_FRAME_COUNTER_H
#define FEM_ENGINE_FRAME_COUNTER_H


#include <iostream>
#include <chrono>

class FrameRateCounter {
public:
    FrameRateCounter() : frameCount(0), lastTime(std::chrono::high_resolution_clock::now()) {}

    void update() {
        frameCount++;
        auto currentTime = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(currentTime - lastTime).count();

        if (elapsed >= 1) { // Every second
            std::cout << "FPS: " << frameCount / elapsed << std::endl;
            frameCount = 0;
            lastTime = currentTime;
        }
    }

private:
    int frameCount;
    std::chrono::time_point<std::chrono::high_resolution_clock> lastTime;
};

#endif //FEM_ENGINE_FRAME_COUNTER_H
