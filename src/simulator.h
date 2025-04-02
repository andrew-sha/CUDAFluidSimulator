class Simulator {
private:
    int numParticles;
    float* position;
    float* velocity;

    float* cudaDevicePosition;
    float* cudaDeviceVelocity;

public:
    Simulator();
    virtual ~Simulator();

    const float* getPosition();

    void setup();

    void loadScene(SceneName name);

    void allocOutputImage(int width, int height);

    void clearImage();

    // simulates a single timestep using SPH
    void simulate();
};
