struct Settings {
    size_t numParticles;
    float h;
    float boxDim;
};

class Simulator {
private:
    float* position;

    float* cudaDevicePosition;
    float* cudaDeviceVelocity;

public:
    const Settings *settings;

    Simulator(Settings* settings);
    virtual ~Simulator();

    void setup();

    const float* getPosition();

    // simulates a single timestep using SPH
    void simulate();
};
