#include <iostream>

#include "cuda_runtime.h"

struct Settings {
    size_t numParticles;
    float h;
    float boxDim;
    float numCellsPerDim;
};

/**
 * @brief Particle struct
 */
struct Particle {
    float3 position, velocity;
    float density, pressure, force;

    struct Particle *next;

    Particle(float3 pos) {
        position = pos;

        velocity = {0.f, 0.f, 0.f};
        density = pressure = force = 0.f;

        next = NULL;
    }

    void display() {
        std::cout << "(" << position.x << ", " << position.y << ", "
                  << position.z << ")" << std::endl;
    }
};

class Simulator {
  private:
    float *position;
    Particle **neighborGrid;
    Particle *particles;

  public:
    const Settings *settings;

    Simulator(Settings *settings);
    virtual ~Simulator();

    void setup();

    const float *getPosition();

    // simulates a single timestep using SPH
    void simulate();
};
