#include "platformgl.h"
#include "simulator.h"

void startVisualization(Simulator *simulator);

int main(int argc, char **argv) {
    // Hardcode some default sim parameters
    // Maybe in the future we can configure these as CLI options
    Settings settings = {10000, 5.f, 15.f, 3};

    Simulator *simulator = new Simulator(&settings);
    simulator->setup();
    glutInit(&argc, argv);
    startVisualization(simulator);

    return 0;
}
