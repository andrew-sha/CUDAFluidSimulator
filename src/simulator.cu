#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Kernels








// Class methods
Simulator::Simulator() {
    numberOfCircles = 0;
    position = NULL;
    velocity = NULL;

    cudaDevicePosition = NULL;
    cudaDeviceVelocity = NULL;
}

Simulator::~Simulator() {
    if (position) {
        delete [] position;
        delete [] velocity;
    }

    if (cudaDevicePosition) {
        cudaFree(cudaDevicePosition);
        cudaFree(cudaDeviceVelocity);
    }
}

const float*
Simulator::getPosition() {

    printf("Copying position data from device\n");

    // Have host_positions as a class attribute which you memcpy into
    float *host_positions = (float *)malloc(sizeof(float) * 3 * numberOfCircles);

    cudaMemcpy(host_positions,
               cudaDevicePosition,
               sizeof(float) * 3 * numberOfCircles,
               cudaMemcpyDeviceToHost);

    return host_positions;
}