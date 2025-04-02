#include "simulator.h"
#include

void startVisualization(Simulator* simulator);

int main(int argc, char** argv) {
    Simulator simulator = new Simulator();
    simulator.setup();
    startVisualization(simulator);
}