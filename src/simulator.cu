#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdio.h>

#include "simulator.h"

#define MAX_THREADS_PER_BLOCK (128)
#define PUSH_STRENGTH (5.f)
#define EPS_F (1e-4f)

extern bool mouseClicked;
extern int2 clickCoords;

__constant__ Settings deviceSettings;

// Device helper functions
// Print neighbor grid
__device__ void printGridList(Particle **neighborGrid) {
    int total = 0;
    int numCells = pow(deviceSettings.numCellsPerDim, 3);
    for (int i = 0; i < numCells; i++) {
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

    printf("Found total of %d particles\n", total);
}

// Lock-free linked list head insertion using compare-and-swap (CAS)
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
__device__ int3 getGridCell(float3 position) {
    int3 gridCell;
    gridCell.x = (int)(position.x / deviceSettings.h);
    if (gridCell.x < 0 || gridCell.x >= deviceSettings.numCellsPerDim) {
        printf("OOB particle: x = %d\n", gridCell.x);
        printf("(%f, %f, %f)\n", position.x, position.y, position.z);
    }
    gridCell.y = (int)(position.y / deviceSettings.h);
    if (gridCell.y < 0 || gridCell.y >= deviceSettings.numCellsPerDim) {
        printf("OOB particle: y = %d\n", gridCell.y);
        printf("(%f, %f, %f)\n", position.x, position.y, position.z);
    }
    gridCell.z = (int)(position.z / deviceSettings.h);
    if (gridCell.z < 0 || gridCell.z >= deviceSettings.numCellsPerDim) {
        printf("OOB particle: z = %d\n", gridCell.z);
        printf("(%f, %f, %f)\n", position.x, position.y, position.z);
    }

    return gridCell;
}

// Convert 3D coordinates of neighbor grid cell to corresponding array index
__device__ int flattenGridCoord(int3 coord) {
    return coord.x + coord.y * deviceSettings.numCellsPerDim +
           coord.z * deviceSettings.numCellsPerDim *
               deviceSettings.numCellsPerDim;
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

    float diff = h2 - dist2;
    return deviceSettings.d_kernel_coeff * diff * diff * diff;
}

// Smoothing kernel for pressure force updates
__device__ float3 pressureKernel(Particle *pi, Particle *pj) {
    float dx = pi->position.x - pj->position.x;
    float dy = pi->position.y - pj->position.y;
    float dz = pi->position.z - pj->position.z;
    float dist2 = dx * dx + dy * dy + dz * dz;

    if (dist2 > deviceSettings.h * deviceSettings.h) {
        return make_float3(0.f, 0.f, 0.f);
    }

    float dist = sqrtf(dist2);
    if (dist < EPS_F)
        return make_float3(0.f, 0.f, 0.f);

    float scale = (-deviceSettings.v_kernel_coeff) * (deviceSettings.h - dist) *
                  (deviceSettings.h - dist) / dist;

    return make_float3(dx * scale, dy * scale, dz * scale);
}

// Smoothing kernel for viscosity force updates
__device__ float viscosityKernel(Particle *pi, Particle *pj) {
    float dx = pi->position.x - pj->position.x;
    float dy = pi->position.y - pj->position.y;
    float dz = pi->position.z - pj->position.z;
    float dist = sqrtf(dx * dx + dy * dy + dz * dz);

    if ((dist > deviceSettings.h) || (dist < EPS_F)) {
        return 0.f;
    }

    return deviceSettings.v_kernel_coeff * (deviceSettings.h - dist);
}

// Kernels
__global__ void kernelBuildGrid(Particle *particles, Particle **neighborGrid) {
    int pIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (pIdx >= deviceSettings.numParticles) {
        return;
    }

    Particle *particle = &particles[pIdx];
    particle->next = NULL;

    int3 cell = getGridCell(particle->position);
    int listIdx = flattenGridCoord(cell);

    insertList(particle, &neighborGrid[listIdx]);
}

__global__ void kernelUpdatePressureAndDensity(Particle *particles,
                                               Particle **neighborGrid) {
    int pIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (pIdx >= deviceSettings.numParticles) {
        return;
    }

    Particle *particle = &particles[pIdx];

    int3 cell = getGridCell(particle->position);

    // Update density for each particle
    particle->density = 0.f;
    for (int dz = -1; dz < 2; dz++) {
        int searchZ = cell.z + dz;
        if (searchZ < 0 || searchZ >= deviceSettings.numCellsPerDim)
            continue;
        for (int dy = -1; dy < 2; dy++) {
            int searchY = cell.y + dy;
            if (searchY < 0 || searchY >= deviceSettings.numCellsPerDim)
                continue;
            for (int dx = -1; dx < 2; dx++) {
                int searchX = cell.x + dx;
                if (searchX < 0 || searchX >= deviceSettings.numCellsPerDim)
                    continue;
                int neighborCellIdx =
                    flattenGridCoord(make_int3(searchX, searchY, searchZ));
                Particle *neighbor = neighborGrid[neighborCellIdx];
                while (neighbor != NULL) {
                    particle->density +=
                        MASS * densityKernel(particle, neighbor);
                    neighbor = neighbor->next;
                }
            }
        }
    }
    particle->density = fmaxf(particle->density, EPS_F);
    // Update pressure for each particle
    particle->pressure =
        fmaxf(0.f, GAS_CONSTANT * (particle->density - REST_DENSITY));
}

__global__ void kernelUpdateForces(Particle *particles,
                                   Particle **neighborGrid) {
    int pIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (pIdx >= deviceSettings.numParticles) {
        return;
    }

    Particle *particle = &particles[pIdx];

    int3 cell = getGridCell(particle->position);
    particle->force.x = 0.f;
    particle->force.y = 0.f;
    particle->force.z = 0.f;

    // Update force for each particle
    for (int dz = -1; dz < 2; dz++) {
        int searchZ = cell.z + dz;
        if (searchZ < 0 || searchZ >= deviceSettings.numCellsPerDim)
            continue;
        for (int dy = -1; dy < 2; dy++) {
            int searchY = cell.y + dy;
            if (searchY < 0 || searchY >= deviceSettings.numCellsPerDim)
                continue;
            for (int dx = -1; dx < 2; dx++) {
                int searchX = cell.x + dx;
                if (searchX < 0 || searchX >= deviceSettings.numCellsPerDim)
                    continue;
                int neighborCellIdx =
                    flattenGridCoord(make_int3(searchX, searchY, searchZ));
                Particle *neighbor = neighborGrid[neighborCellIdx];
                while (neighbor != NULL) {
                    // Calculate pressure force
                    float fPressure =
                        -MASS * (particle->pressure + neighbor->pressure) /
                        (2.f * neighbor->density);
                    float3 kern1 = pressureKernel(particle, neighbor);
                    kern1.x *= fPressure;
                    kern1.y *= fPressure;
                    kern1.z *= fPressure;
                    particle->force.x += kern1.x;
                    particle->force.y += kern1.y;
                    particle->force.z += kern1.z;

                    // Calculate viscosity force
                    float3 dv = make_float3(
                        neighbor->velocity.x - particle->velocity.x,
                        neighbor->velocity.y - particle->velocity.y,
                        neighbor->velocity.z - particle->velocity.z);
                    float fViscosity = VISCOSITY * MASS *
                                       viscosityKernel(particle, neighbor) /
                                       neighbor->density;
                    dv.x *= fViscosity;
                    dv.y *= fViscosity;
                    dv.z *= fViscosity;
                    particle->force.x += dv.x;
                    particle->force.y += dv.y;
                    particle->force.z += dv.z;

                    neighbor = neighbor->next;
                }
            }
        }
    }
}

__global__ void kernelUpdatePositions(Particle *particles,
                                      float3 *devicePosition) {
    int pIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (pIdx >= deviceSettings.numParticles) {
        return;
    }

    Particle *particle = &particles[pIdx];
    float timestep = deviceSettings.timestep;

    if (!isfinite(particle->force.x) || !isfinite(particle->velocity.x)) {
        printf("Bad force/velocity at particle %d: fx=%f, vx=%f\n", pIdx,
               particle->force.x, particle->velocity.x);
    }
    if (!isfinite(particle->force.y) || !isfinite(particle->velocity.y)) {
        printf("Bad force/velocity at particle %d: fy=%f, vy=%f\n", pIdx,
               particle->force.y, particle->velocity.y);
    }
    if (!isfinite(particle->force.z) || !isfinite(particle->velocity.z)) {
        printf("Bad force/velocity at particle %d: fz=%f, vz=%f\n", pIdx,
               particle->force.z, particle->velocity.z);
    }

    particle->velocity.x += timestep * particle->force.x / particle->density;
    particle->velocity.y +=
        timestep * (particle->force.y / particle->density + GRAVITY);
    particle->velocity.z += timestep * particle->force.z / particle->density;

    particle->position.x += timestep * particle->velocity.x;
    particle->position.y += timestep * particle->velocity.y;
    particle->position.z += timestep * particle->velocity.z;

    // Handle boundary collisions
    if (particle->position.x < deviceSettings.h) {
        particle->position.x = deviceSettings.h;
        particle->velocity.x *= -ELASTICITY;
    } else if (particle->position.x >
               deviceSettings.boxDim - deviceSettings.h) {
        particle->position.x = deviceSettings.boxDim - deviceSettings.h;
        particle->velocity.x *= -ELASTICITY;
    }

    if (particle->position.y < deviceSettings.h) {
        particle->position.y = deviceSettings.h;
        particle->velocity.y *= -ELASTICITY;
    } else if (particle->position.y >
               deviceSettings.boxDim - deviceSettings.h) {
        particle->position.y = deviceSettings.boxDim - deviceSettings.h;
        particle->velocity.y *= -ELASTICITY;
    }

    if (particle->position.z < deviceSettings.h) {
        particle->position.z = deviceSettings.h;
        particle->velocity.z *= -ELASTICITY;
    } else if (particle->position.z >
               deviceSettings.boxDim - deviceSettings.h) {
        particle->position.z = deviceSettings.boxDim - deviceSettings.h;
        particle->velocity.z *= -ELASTICITY;
    }

    if (fabs(particle->velocity.x) < EPS_F) {
        particle->velocity.x = 0;
    }
    if (fabs(particle->velocity.y) < EPS_F) {
        particle->velocity.y = 0;
    }
    if (fabs(particle->velocity.z) < EPS_F) {
        particle->velocity.z = 0;
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

// Induce velocity on mouse click
__global__ void kernelMoveParticles(Particle **neighborGrid, int2 mouse_pos) {
    // Normalize the mouse positions to the box's size
    float x =
        ((float)(mouse_pos.x - BOX_MIN_X) / (float)(BOX_MAX_X - BOX_MIN_X)) *
        deviceSettings.boxDim;
    float y =
        ((float)(mouse_pos.y - BOX_MIN_Y) / (float)(BOX_MAX_Y - BOX_MIN_Y)) *
        deviceSettings.boxDim;
    float z = (float)threadIdx.x * deviceSettings.h;

    int3 cell = getGridCell(make_float3(x, y, z));
    cell.y = deviceSettings.numCellsPerDim - cell.y;

    for (int dy = -2; dy < 3; dy++) {
        int searchY = cell.y + dy;
        if (searchY < 0 || searchY >= deviceSettings.numCellsPerDim)
            continue;

        for (int dx = -2; dx < 3; dx++) {
            int searchX = cell.x + dx;
            if (searchX < 0 || searchX >= deviceSettings.numCellsPerDim)
                continue;
            int neighborCellIdx =
                flattenGridCoord(make_int3(searchX, searchY, cell.z));
            Particle *neighbor = neighborGrid[neighborCellIdx];

            while (neighbor != NULL) {
                // update the velocity of each particle within the cell
                if (dx != 0)
                    neighbor->velocity.x += (1.f / dx) * PUSH_STRENGTH;
                if (dy != 0)
                    neighbor->velocity.y += (1.f / dy) * PUSH_STRENGTH;
                if (dx == 0 && dy == 0)
                    neighbor->velocity.z -= PUSH_STRENGTH;

                neighbor = neighbor->next;
            }
        }
    }
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
    return position;
}

void Simulator::setup() {
    // Initialize device data structures
    int neighborGridDim = settings->numCellsPerDim;
    cudaMalloc(&neighborGrid, neighborGridDim * neighborGridDim *
                                  neighborGridDim * sizeof(Particle *));
    cudaMalloc(&particles, settings->numParticles * sizeof(Particle));
    cudaMalloc(&devicePosition, settings->numParticles * sizeof(float3));

    cudaMemset(neighborGrid, 0,
               neighborGridDim * neighborGridDim * neighborGridDim *
                   sizeof(Particle *));
    cudaMemset(particles, 0, settings->numParticles * sizeof(Particle));

    // Initialize particle positions
    Particle *tmpParticles =
        (Particle *)malloc(sizeof(Particle) * settings->numParticles);

    position = (float3 *)malloc(sizeof(float3) * settings->numParticles);

    if (settings->randomInit) {
        for (size_t i = 0; i < settings->numParticles; i++) {
            float x = rand() / (float)RAND_MAX * (settings->boxDim - 2.f) + 1.f;
            float y = rand() / (float)RAND_MAX * (settings->boxDim - 2.f) + 1.f;
            float z = rand() / (float)RAND_MAX * (settings->boxDim - 2.f) + 1.f;

            tmpParticles[i] = Particle(make_float3(x, y, z));
        }
    } else {
        float spacing = 0.9f * settings->h;
        int nx = floor((settings->boxDim - 2 * settings->h) / spacing) + 1;
        int ny = nx, nz = nx;

        int count = 0;
        for (int x = 0; x < nx && count < settings->numParticles; x++) {
            for (int y = 0; y < ny && count < settings->numParticles; y++) {
                for (int z = 0; z < nz && count < settings->numParticles; z++) {
                    tmpParticles[count++] = Particle(make_float3(
                        settings->h + spacing * x, settings->h + spacing * y,
                        settings->h + spacing * z));
                }
            }
        }
    }

    // Copy initialized particles to device
    cudaMemcpy(particles, tmpParticles,
               settings->numParticles * sizeof(Particle),
               cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(deviceSettings, settings, sizeof(Settings));
}

void Simulator::simulate() {
    // 1. Build neighbor grid
    dim3 blockDim(MAX_THREADS_PER_BLOCK);
    dim3 gridDim((settings->numParticles + MAX_THREADS_PER_BLOCK - 1) /
                 MAX_THREADS_PER_BLOCK);

    kernelBuildGrid<<<gridDim, blockDim>>>(particles, neighborGrid);
    cudaDeviceSynchronize();

    // 2. Compute updates
    kernelUpdatePressureAndDensity<<<gridDim, blockDim>>>(particles,
                                                          neighborGrid);
    kernelUpdateForces<<<gridDim, blockDim>>>(particles, neighborGrid);
    kernelUpdatePositions<<<gridDim, blockDim>>>(particles, devicePosition);
    cudaDeviceSynchronize();

    // 3. Copy updated positions to host
    cudaMemcpy(position, devicePosition,
               sizeof(float3) * settings->numParticles, cudaMemcpyDeviceToHost);

    if (mouseClicked) {
        dim3 blockDim2(settings->numCellsPerDim);
        dim3 gridDim2(1);

        // Launch a kernel to update the velocity of the particles
        kernelMoveParticles<<<gridDim2, blockDim2>>>(neighborGrid, clickCoords);
        cudaDeviceSynchronize();
        mouseClicked = false;
    }

    // 4. Reset heads of the neighbor lists
    dim3 resetGridDim(settings->numCellsPerDim, settings->numCellsPerDim,
                      settings->numCellsPerDim);
    dim3 resetBlockDim(1);
    kernelResetGrid<<<resetGridDim, resetBlockDim>>>(neighborGrid);
    cudaDeviceSynchronize();
}

void Simulator::simulateAndTime(Times *times) {
    // 1. Build neighbor grid
    dim3 blockDim(MAX_THREADS_PER_BLOCK);
    dim3 gridDim((settings->numParticles + MAX_THREADS_PER_BLOCK - 1) /
                 MAX_THREADS_PER_BLOCK);

    auto buildGridStart = std::chrono::steady_clock::now();

    kernelBuildGrid<<<gridDim, blockDim>>>(particles, neighborGrid);
    cudaDeviceSynchronize();

    times->buildGrid +=
        std::chrono::duration_cast<std::chrono::duration<double>>(
            std::chrono::steady_clock::now() - buildGridStart)
            .count();
    // 2. Compute updates
    auto sphUpdateStart = std::chrono::steady_clock::now();

    // 2. Compute updates
    kernelUpdatePressureAndDensity<<<gridDim, blockDim>>>(particles,
                                                          neighborGrid);
    kernelUpdateForces<<<gridDim, blockDim>>>(particles, neighborGrid);
    kernelUpdatePositions<<<gridDim, blockDim>>>(particles, devicePosition);
    cudaDeviceSynchronize();

    times->sphUpdate +=
        std::chrono::duration_cast<std::chrono::duration<double>>(
            std::chrono::steady_clock::now() - sphUpdateStart)
            .count();

    // 3. Copy updated positions to host
    auto memcpyStart = std::chrono::steady_clock::now();

    cudaMemcpy(position, devicePosition,
               sizeof(float3) * settings->numParticles, cudaMemcpyDeviceToHost);

    times->memcpy += std::chrono::duration_cast<std::chrono::duration<double>>(
                         std::chrono::steady_clock::now() - memcpyStart)
                         .count();
    // 4. Reset heads of the neighbor lists
    dim3 resetGridDim(settings->numCellsPerDim, settings->numCellsPerDim,
                      settings->numCellsPerDim);
    dim3 resetBlockDim(1);
    kernelResetGrid<<<resetGridDim, resetBlockDim>>>(neighborGrid);
    cudaDeviceSynchronize();

    times->iters += 1;
}
