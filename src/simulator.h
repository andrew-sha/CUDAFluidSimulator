#include <iostream>
#include <stdio.h>

#include "cuda_runtime.h"

struct Settings {
    size_t numParticles;
    float h;
    float boxDim;
    float numCellsPerDim;
    float timestep;
};

/**
 * @brief Particle struct
 */
struct Particle {
    float3 position, velocity, force;
    float density, pressure;

    struct Particle *next;

    Particle(float3 pos) {
        position = pos;

        velocity = force = {0.f, 0.f, 0.f};
        density = pressure = 0.f;

        next = NULL;
    }

    void display() {
        printf("(%f, %f, %f)\n", position.x, position.y, position.z);
    }
};

class Simulator {
  private:
    float3 *position;
    float3 *devicePosition;

    Particle **neighborGrid;
    Particle *particles;

  public:
    const Settings *settings;

    Simulator(Settings *settings);
    virtual ~Simulator();

    void setup();

    const float3 *getPosition();

    // simulates a single timestep using SPH
    void simulate();
};
