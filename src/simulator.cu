#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <stdio.h>

#include "simulator.h"

#define MAX_THREADS_PER_BLOCK (1024)

#define PI 3.14159265f
#define MASS 0.02f
#define GAS_CONSTANT 1.f  // corresponds to temperature
#define REST_DENSITY 1000.f
#define VISCOSITY 3.5f
#define GRAVITY -9.8f
#define ELASTICITY 0.5f;

__constant__ Settings deviceSettings;

// Device helper functions
// Print neighbor grid
__device__ void printGridList(Particle **neighborGrid) {
    int total = 0;
    for (int i = 0; i < pow(deviceSettings.numCellsPerDim, 3); i++) {
        Particle *head = neighborGrid[i];
        printf("LIST %d:\n", i);
        printf("======================\n");

        while (head != NULL) {
            total++;
            printf("(%f, %f, %f)\n", head->position.x, head->position.y,
                   head->position.z);
            head = head->next;
        }

        printf("\n");
    }

    printf("Found total of %d elements\n", total);
}

// LOCK FREE list insertion
__device__ void insertList(Particle *particle, Particle **head) {
    Particle *oldHead = *head;
    particle->next = oldHead;

    while (atomicCAS((unsigned long long int *)head,
                     (unsigned long long int)oldHead,
                     (unsigned long long int)particle) !=
           (unsigned long long int)oldHead) {
        oldHead = *head;
        particle->next = oldHead;
    }
}

// Return 3D coordinates of neighbor grid cell a particle belongs to
__device__ int3 getGridCell(Particle *particle) {
    // TODO Compute cell position -- segfault possible
    int3 gridCell;
    gridCell.x = (int)(particle->position.x / deviceSettings.h);
    gridCell.y = (int)(particle->position.y / deviceSettings.h);
    gridCell.z = (int)(particle->position.z / deviceSettings.h);

    return gridCell;
}

// Convert 3D coordinates of neighbor grid cell to corresponding array index
__device__ int flattenGridCoord(int3 coord) {
    return coord.x + coord.y * deviceSettings.numCellsPerDim + coord.z * deviceSettings.numCellsPerDim * deviceSettings.numCellsPerDim;
}

// Smoothing kernel for density updates
__device__ float densityKernel(Particle *pi, Particle *pj) {
    float dx = pi->position.x - pj->position.x;
    float dy = pi->position.y - pj->position.y;
    float dz = pi->position.z - pj->position.z;
    float dist2 = dx * dx + dy * dy + dz * dz;
    float h2 = deviceSettings.h * deviceSettings.h;
    
    if (dist2 > h2) {
        return 0.f;
    }

    return 315.f/(64.f * PI * pow(deviceSettings.h, 9)) * pow(h2 - dist2, 3);
}

// Smoothing kernel for pressure force updates
__device__ float3 pressureKernel(Particle *pi, Particle *pj) {
    float dx = pi->position.x - pj->position.x;
    float dy = pi->position.y - pj->position.y;
    float dz = pi->position.z - pj->position.z;
    float dist = sqrtf(dx * dx + dy * dy + dz * dz);


    if (dist > deviceSettings.h || dist == 0.f) {
        return make_float3(0.f, 0.f, 0.f);
    }

    float3 dir = make_float3(dx, dy, dz);
    float k = -45.f/(PI * pow(deviceSettings.h, 6)) * pow(deviceSettings.h - dist, 2);
    k /= dist;
    dir.x *= k;
    dir.y *= k;
    dir.z *= k;

    return dir;
}

// Smoothing kernel for viscosity force updates
__device__ float viscosityKernel(Particle *pi, Particle *pj) {
    float dx = pi->position.x - pj->position.x;
    float dy = pi->position.y - pj->position.y;
    float dz = pi->position.z - pj->position.z;
    float dist = sqrtf(dx * dx + dy * dy + dz * dz);

    if (dist > deviceSettings.h) {
        return 0.f;
    }

    return 45.f/(PI * pow(deviceSettings.h, 6)) * (deviceSettings.h - dist);
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
    int3 cell = getGridCell(particle);

    int listIdx = flattenGridCoord(cell);

    // 3. append it to the appropriate list using lock free
    insertList(particle, &neighborGrid[listIdx]);

    __syncthreads();
    if (pIdx == 0) {
        printGridList(neighborGrid);
    }
}

__global__ void kernelSPHUpdate(Particle *particles, Particle **neighborGrid, float3 *devicePosition) {
    int pIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (pIdx >= deviceSettings.numParticles) {
        return;
    }

    Particle *particle = &particles[pIdx];

    int3 cell = getGridCell(particle);

    // Update density for each particle
    for (int dz = -1; dz < 2; dz++) {
        int searchZ = cell.z + dz;
        if (searchZ < 0 || searchZ > deviceSettings.numCellsPerDim) continue;
        for (int dy = -1; dy < 2; dy++) {
            int searchY = cell.y + dy;
            if (searchY < 0 || searchY > deviceSettings.numCellsPerDim) continue;
            for (int dx = -1; dx < 2; dx++) {
                int searchX = cell.x + dx;
                if (searchX < 0 || searchX > deviceSettings.numCellsPerDim) continue;
                int neighborCellIdx = flattenGridCoord(make_int3(searchX, searchY, searchZ));
                Particle *neighbor = neighborGrid[neighborCellIdx];
                while (neighbor != NULL) {
                    particle->density += MASS * densityKernel(particle, neighbor);
                }
            }
        }
    }

    // Update pressure for each particle
    particle->pressure = GAS_CONSTANT * (particle->density - REST_DENSITY);

    __syncthreads();

    // Update force for each particle
    for (int dz = -1; dz < 2; dz++) {
        int searchZ = cell.z + dz;
        if (searchZ < 0 || searchZ > deviceSettings.numCellsPerDim) continue;
        for (int dy = -1; dy < 2; dy++) {
            int searchY = cell.y + dy;
            if (searchY < 0 || searchY > deviceSettings.numCellsPerDim) continue;
            for (int dx = -1; dx < 2; dx++) {
                int searchX = cell.x + dx;
                if (searchX < 0 || searchX > deviceSettings.numCellsPerDim) continue;
                int neighborCellIdx = flattenGridCoord(make_int3(searchX, searchY, searchZ));
                Particle *neighbor = neighborGrid[neighborCellIdx];
                while (neighbor != NULL) {
                    // Calculate pressure force
                    float fPressure = -MASS * (particle->pressure + neighbor->pressure) / (2.f * neighbor->density);
                    float3 kern1 = pressureKernel(particle, neighbor);
                    kern1.x *= fPressure;
                    kern1.y *= fPressure;
                    kern1.z *= fPressure;
                    particle->force.x += kern1.x;
                    particle->force.y += kern1.y;
                    particle->force.z += kern1.z;

                    // Calculate viscosity force
                    float3 dv = make_float3(neighbor->velocity.x - particle->velocity.x, neighbor->velocity.y - particle->velocity.y, neighbor->velocity.z - particle->velocity.z);
                    float fViscosity = VISCOSITY * MASS * viscosityKernel(particle, neighbor) / neighbor->density;
                    dv.x *= fViscosity;
                    dv.y *= fViscosity;
                    dv.z *= fViscosity;
                    particle->force.x += dv.x;
                    particle->force.y += dv.y;
                    particle->force.z += dv.z;

                    // Calculate gravitational force
                    // Should this impact the y or z direction?
                    particle->force.z += MASS * GRAVITY;
                }
            }
        }
    }

    __syncthreads();

    // Update position for each particle
    float timestep = deviceSettings.timestep;

    particle->velocity.x += timestep * particle->force.x / MASS;
    particle->velocity.y += timestep * particle->force.y / MASS;
    particle->velocity.z += timestep * particle->force.z / MASS;

    particle->position.x += particle->velocity.x * timestep;
    particle->position.y += particle->velocity.y * timestep;
    particle->position.z += particle->velocity.z * timestep;
    
    // Handle boundary collisions
    if (particle->position.x < deviceSettings.h) {
        particle->position.x = deviceSettings.h;
        particle->velocity.x *= -ELASTICITY;
    } else if (particle->position.x > deviceSettings.boxDim - deviceSettings.h) {
        particle->position.x = deviceSettings.boxDim - deviceSettings.h;
        particle->velocity.x *= -ELASTICITY;
    } else if (particle->position.y < deviceSettings.h) {
        particle->position.y = deviceSettings.h;
        particle->velocity.y *= -ELASTICITY;
    } else if (particle->position.y > deviceSettings.boxDim - deviceSettings.h) {
        particle->position.y = deviceSettings.boxDim - deviceSettings.h;
        particle->velocity.y *= -ELASTICITY;
    } else if (particle->position.z < deviceSettings.h) {
        particle->position.z = deviceSettings.h;
        particle->velocity.z *= -ELASTICITY;
    } else if (particle->position.z > deviceSettings.boxDim - deviceSettings.h) {
        particle->position.z = deviceSettings.boxDim - deviceSettings.h;
        particle->velocity.z *= -ELASTICITY;
    }

    // Write updated positions  
    devicePosition[pIdx] = particle->position;
}

// Reset the list heads
__global__ void kernelResetGrid(Particle **neighborGrid) {
    int listIdx = blockIdx.x + blockIdx.y * gridDim.y +
                    blockIdx.z * gridDim.z * gridDim.z;

    neighborGrid[listIdx] = NULL;
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
    kernelSPHUpdate<<<gridDim, blockDim>>>(particles, neighborGrid, devicePosition);

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
