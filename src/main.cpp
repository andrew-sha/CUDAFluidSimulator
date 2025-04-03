#include "platformgl.h"
#include "simulator.h"

void startVisualization(Simulator* simulator);

int main(int argc, char** argv) {
    // Hardcode some default sim parameters
    // Maybe in the future we can configure these as CLI options
    Settings settings = {5, .15f, 10.f};

    Simulator* simulator = new Simulator(&settings);
    //simulator->setup();
    glutInit(&argc, argv);
    startVisualization(simulator);
}