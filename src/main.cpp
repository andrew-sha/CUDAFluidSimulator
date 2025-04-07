#include <unistd.h>

#include <iostream>
#include <string>

#include "platformgl.h"
#include "simulator.h"

void startVisualization(Simulator *simulator);

// TODO
void usage() {
    printf("Program Options:\n");
    printf("  -n  <NUM_PARTICLES>    Number of particles to simulate\n");
    printf("  -i  <random/grid>      Initialization mode: random or grid\n");
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
                if (!(std::string(optarg) == "random" || std::string(optarg) == "grid")) {
                    break;
                }
                randomInit = (std::string(optarg) == "random");
                break;
            case 'm':
                if (!(std::string(optarg) == "time" || std::string(optarg) == "free")) {
                    break;
                }
                benchmark = (std::string(optarg) == "time");
                break;
            case '?':
                usage();
                return 1;
        }
    }

    Settings settings = {randomInit, numParticles, .1f, 10.f, 100, .03};

    Simulator *simulator = new Simulator(&settings);
    simulator->setup();
    glutInit(&argc, argv);
    startVisualization(simulator);

    return 0;
}
