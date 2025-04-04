#include <iostream>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>


#include "simulator.h"

// Kernels






// Class methods
Simulator::Simulator(Settings* settings) : settings(settings) {
    position = NULL;

    cudaDevicePosition = NULL;
    cudaDeviceVelocity = NULL;
}

Simulator::~Simulator() {
    if (position) {
        delete [] position;
    }

    if (cudaDevicePosition) {
        cudaFree(cudaDevicePosition);
        cudaFree(cudaDeviceVelocity);
    }
}

const float*
Simulator::getPosition() {

    std::cout << "Copying position data from device" << std::endl;

    // cudaMemcpy(position,
    //            cudaDevicePosition,
    //            sizeof(float) * 3 * settings->numParticles,
    //            cudaMemcpyDeviceToHost);

    return position;
}

void
Simulator::setup() {
    if (position == NULL) {
        position = (float *)malloc(sizeof(float) * 3 * settings->numParticles);
    }

    // set random initial particle positions
    for (size_t i = 0; i < settings->numParticles; i++) {
        size_t particleIdx = 3 * i;
        position[particleIdx] = rand() / (float)RAND_MAX * settings->boxDim;
        position[particleIdx+1] = rand() / (float)RAND_MAX * settings->boxDim;
        position[particleIdx+2] = rand() / (float)RAND_MAX * settings->boxDim;
        std::cout << i << ": (" << position[particleIdx] << ", " << position[particleIdx+1] << ", " << position[particleIdx+2] << ")" << std::endl;
    }
}

void Simulator::simulate() {
    return;
}