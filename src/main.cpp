#include <unistd.h>

#include <cmath>
#include <iostream>
#include <string>

#include "platformgl.h"
#include "simulator.h"

void startVisualization(Simulator *simulator);

void usage() {
    printf("Program Options:\n");
    printf("  -n  <NUM_PARTICLES>    Number of particles to simulate\n");
    printf("  -i  <random/grid>      Initialization mode: random or grid\n");
    printf("  -m  <free/time>        Execution mode: free or timed\n");
    printf("  -?                     This message\n");
}

int main(int argc, char **argv) {
    int numParticles = 1000;
    bool randomInit = false;
    bool benchmark = true;
    int opt;

    while ((opt = getopt(argc, argv, "n:i:m:?")) != -1) {
        switch (opt) {
        case 'n':
            numParticles = std::stoi(optarg);
            break;
        case 'i':
            if (!(std::string(optarg) == "random" ||
                  std::string(optarg) == "grid")) {
                std::cout << "Invalid argument for option -i: " << optarg
                          << std::endl;
                usage();
                return 1;
            }
            randomInit = (std::string(optarg) == "random");
            break;
        case 'm':
            if (!(std::string(optarg) == "time" ||
                  std::string(optarg) == "free")) {
                std::cout << "Invalid argument for option -m: " << optarg
                          << std::endl;
                usage();
                return 1;
            }
            benchmark = (std::string(optarg) == "time");
            break;
        case '?':
            usage();
            return 1;
        }
    }

    float h = .1f;
    float h_pow_6 = pow(h, 6);
    float h_pow_9 = pow(h, 9);
    float v_kernel_coeff = 45.f / (PI * h_pow_6);
    float d_kernel_coeff = 315.f / (64.f * PI * h_pow_9);

    Settings settings = {randomInit,     numParticles, h,   v_kernel_coeff,
                         d_kernel_coeff, 10.f,         100, .01};

    Simulator *simulator = new Simulator(&settings);
    simulator->setup();

    if (benchmark) {
        int numIters = 100;
        Times times;

        for (int i = 0; i < numIters; i++) {
            simulator->simulateAndTime(&times);
        }

        displayTimes(&times);
    } else {
        glutInit(&argc, argv);
        startVisualization(simulator);
    }

    return 0;
}
