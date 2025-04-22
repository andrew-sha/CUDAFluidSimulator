#include <stdio.h>

#include "cuda_runtime.h"
#include "times.h"

#define PI 3.14159265f
#define MASS 0.02f
#define GAS_CONSTANT 1.f // corresponds to temperature
#define REST_DENSITY 1000.f
#define VISCOSITY 1.f
#define GRAVITY -9.8f
#define ELASTICITY 0.5f;

// Confines of the sad water box
#define BOX_MAX_X (600)
#define BOX_MIN_X (200)
#define BOX_MAX_Y (450)
#define BOX_MIN_Y (150)

struct Settings {
    bool randomInit;
    int numParticles;
    float h;

    // pre-computed constants
    float v_kernel_coeff;
    float d_kernel_coeff;

    float boxDim;
    float numCellsPerDim;
    float timestep;
};

struct Particle {
    float3 position, velocity, force;
    float density, pressure;

    unsigned int cellID;

    __host__ __device__ Particle() {
        position = velocity = force = {0.f, 0.f, 0.f};
        density = pressure = 0.f;
    }

    __host__ __device__ Particle(float3 pos) {
        position = pos;

        velocity = force = {0.f, 0.f, 0.f};
        density = pressure = 0.f;
    }
};

struct ParticleComp {
    __host__ __device__ bool operator()(const Particle &pi,
                                        const Particle &pj) const {
        return pi.cellID < pj.cellID;
    }
};

class Simulator {
  private:
    float3 *position;
    float3 *devicePosition;

    int *neighborGrid;

    Particle *particles;

  public:
    const Settings *settings;

    Simulator(Settings *settings);
    virtual ~Simulator();

    void setup();

    const float3 *getPosition();

    void buildNeighborGrid();
    void simulate();
    void simulateAndTime(Times *times);
};
