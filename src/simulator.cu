#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <stdio.h>

#include "simulator.h"

#define MAX_THREADS_PER_BLOCK (1024)

__constant__ Settings deviceSettings;

// GPU helper functions
// Print the linked list
__device__ void print_list(Particle **neighborGrid) {
    int total_count = 0;
    for (int i = 0; i < pow(deviceSettings.numCellsPerDim, 3); i++) {
        Particle *head = neighborGrid[i];
        printf("LIST %d:\n", i);
        printf("======================\n");

        while (head != NULL) {
            total_count++;
            printf("(%f, %f, %f)\n", head->position.x, head->position.y,
                   head->position.z);
            head = head->next;
        }

        printf("\n");
    }

    printf("Found total of %d elements\n", total_count);
}

// LOCK FREE list insertion
__device__ void insert_list(Particle *particle, Particle **head) {
    Particle *old_head = *head;
    particle->next = old_head;

    while (atomicCAS((unsigned long long int *)head,
                     (unsigned long long int)old_head,
                     (unsigned long long int)particle) !=
           (unsigned long long int)old_head) {
        old_head = *head;
        particle->next = old_head;
    }
}

// Kernels
__global__ void kernelBuildGrid(Particle *particles, Particle **neighborGrid) {
    // 1. each thread calculates the index of the particle
    int pIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (pIdx >= deviceSettings.numParticles) {
        return;
    }

    // 2. calc coordinates of the cell that the particle belongs to
    Particle *particle = &particles[pIdx];

    // TODO Compute cell position -- segfault possible
    int3 cell;
    cell.x = (int)(particle->position.x / deviceSettings.h);
    cell.y = (int)(particle->position.y / deviceSettings.h);
    cell.z = (int)(particle->position.z / deviceSettings.h);

    int listIndex =
        cell.x + cell.y * deviceSettings.numCellsPerDim +
        cell.z * deviceSettings.numCellsPerDim * deviceSettings.numCellsPerDim;

    // 3. append it to the appropriate list using lock free
    insert_list(particle, &neighborGrid[listIndex]);

    __syncthreads();
    if (pIdx == 0) {
        print_list(neighborGrid);
    }
}

// Reset the list heads
__global__ void kernelResetGrid(Particle **neighborGrid) {
    int listIndex = blockIdx.x + blockIdx.y * gridDim.y +
                    blockIdx.z * gridDim.z * gridDim.z;

    neighborGrid[listIndex] = NULL;
}

// Class methods
Simulator::Simulator(Settings *settings) : settings(settings) {
    position = NULL;

    neighborGrid = NULL;
    particles = NULL;
}

Simulator::~Simulator() {
    if (position) {
        delete[] position;
    }

    int neighborGridDim = settings->boxDim / settings->h;
    int totalCubes = neighborGridDim * neighborGridDim * neighborGridDim;

    // Free the linked list of particles on device
    if (neighborGrid != NULL) {
        for (int i = 0; i < totalCubes; i++) {
            Particle *prev = NULL;
            Particle *head = neighborGrid[i];
            while (head != NULL) {
                prev = head;
                head = head->next;

                cudaFree(prev);
            }
        }

        // Free the grid on device
        cudaFree(neighborGrid);
    }

    // Free the particles on device
    if (particles != NULL) {
        cudaFree(particles);
        cudaFree(devicePosition);
    }
}

const float3 *Simulator::getPosition() {
    std::cout << "Copying position data from device" << std::endl;
    return position;
}

void Simulator::setup() {
    // create a position array for the device which will be copied back to the
    // host

    int neighborGridDim = settings->boxDim / settings->h;
    cudaMalloc(&neighborGrid, neighborGridDim * neighborGridDim *
                                  neighborGridDim * sizeof(Particle *));
    cudaMalloc(&particles, settings->numParticles * sizeof(Particle));
    cudaMalloc(&devicePosition, settings->numParticles * sizeof(float3));

    // Zero out device memory
    cudaMemset(neighborGrid, 0,
               neighborGridDim * neighborGridDim * neighborGridDim *
                   sizeof(Particle *));
    cudaMemset(particles, 0, settings->numParticles * sizeof(Particle));

    Particle *cpuParticles =
        (Particle *)malloc(sizeof(Particle) * settings->numParticles);

    if (position == NULL) {
        position = (float3 *)malloc(sizeof(float3) * settings->numParticles);
    }

    // set random initial particle positions
    for (size_t i = 0; i < settings->numParticles; i++) {
        float x = rand() / (float)RAND_MAX * settings->boxDim;
        float y = rand() / (float)RAND_MAX * settings->boxDim;
        float z = rand() / (float)RAND_MAX * settings->boxDim;

        cpuParticles[i] = Particle(make_float3(x, y, z));
    }

    // Copy the porticles to device
    cudaMemcpy(particles, cpuParticles,
               settings->numParticles * sizeof(Particle),
               cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(deviceSettings, settings, sizeof(Settings));

    free(cpuParticles);
}

void Simulator::simulate() {
    // 1. Build the grid
    dim3 blockDim(MAX_THREADS_PER_BLOCK);
    dim3 gridDim((settings->numParticles + MAX_THREADS_PER_BLOCK - 1) /
                 MAX_THREADS_PER_BLOCK);

    kernelBuildGrid<<<gridDim, blockDim>>>(particles, neighborGrid);

    // 2. Compute updates

    // 3. Send positions back to host
    cudaMemcpy(position, devicePosition,
               sizeof(float3) * settings->numParticles, cudaMemcpyDeviceToHost);

    // 4. Reset the heads of the linked lists
    dim3 resetGridDim(settings->numCellsPerDim, settings->numCellsPerDim,
                      settings->numCellsPerDim);
    dim3 resetBlockDim(1);
    kernelResetGrid<<<resetGridDim, resetBlockDim>>>(neighborGrid);

    cudaDeviceSynchronize();
    return;
}
