#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdio.h>

#include "simulator.h"

#define MAX_THREADS_PER_BLOCK (128)
#define PUSH_STRENGTH (5.f)
#define CHUNK_COUNT (3)

extern bool mouseClicked;
extern int2 clickCoords;

__constant__ Settings deviceSettings;

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

    return 315.f / (64.f * PI * pow(deviceSettings.h, 9)) * pow(h2 - dist2, 3);
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
    float safeDist = fmaxf(dist, 1e-3f); // safe guard
    float scale = -45.f / (PI * pow(deviceSettings.h, 6)) * pow(deviceSettings.h - dist, 2) / safeDist;

    return make_float3(dx * scale, dy * scale, dz * scale);


    // float dist = sqrtf(dist2);
    // float3 dir = make_float3(dx, dy, dz);
    // float k = -45.f / (PI * pow(deviceSettings.h, 6)) *
    //           pow(deviceSettings.h - dist, 2);
    // k /= dist;
    // dir.x *= k;
    // dir.y *= k;
    // dir.z *= k;

    // return dir;
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

    return 45.f / (PI * pow(deviceSettings.h, 6)) * (deviceSettings.h - dist);
}

// Kernels
__global__ void kernelAssignCellID(Particle *particles) {
    int pIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (pIdx >= deviceSettings.numParticles) {
        return;
    }

    Particle *particle = &particles[pIdx];
    int3 cell = getGridCell(particle->position);
    int flatCellIdx = flattenGridCoord(cell);
    particle->cellID = flatCellIdx;
}

__global__ void kernelPopulateGrid(Particle *particles, int *neighborGrid) {
    int pIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (pIdx >= deviceSettings.numParticles) {
        return;
    }

    int myCellID = particles[pIdx].cellID;
    if (pIdx == 0 || myCellID != particles[pIdx - 1].cellID) {
        neighborGrid[myCellID] = pIdx;
    }
}

__global__ void kernelUpdatePressureAndDensity(Particle *particles,
                                               int *neighborGrid) {
    int pIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int myChunkIdx = blockIdx.x;
    int totalChunks = gridDim.x;

    int startChunkIdx = max(myChunkIdx - (CHUNK_COUNT / 2), 0);

    if (myChunkIdx + (CHUNK_COUNT / 2) >= totalChunks) {
        startChunkIdx = max(0, totalChunks - CHUNK_COUNT);
    }

    int firstParticleIdx = startChunkIdx * MAX_THREADS_PER_BLOCK;

    if (pIdx >= deviceSettings.numParticles) {
        return;
    }

    // Shared array to store the particles related to this block
    __shared__ Particle myParticles[MAX_THREADS_PER_BLOCK * CHUNK_COUNT];

    for (int i = 0; i < CHUNK_COUNT; i++) {
        int particleToLoad = (firstParticleIdx + i * blockDim.x) + threadIdx.x;
        if (particleToLoad >= deviceSettings.numParticles) break;
        myParticles[particleToLoad - firstParticleIdx] = particles[particleToLoad];
    }

    __syncthreads();

    Particle *particle = &myParticles[(myChunkIdx - startChunkIdx) * MAX_THREADS_PER_BLOCK + threadIdx.x];
    int3 cell = getGridCell(particle->position);
    particle->density = 0.f;

    // Update density based on neighbors
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
                int neighborIdx = neighborGrid[neighborCellIdx];
                if (neighborIdx == -1)
                    continue;
                for (int i = neighborIdx; i < deviceSettings.numParticles;
                     i++) {
                    Particle *neighbor = NULL;

                    if ((i >= firstParticleIdx) &&
                        (i < firstParticleIdx +
                                 MAX_THREADS_PER_BLOCK * CHUNK_COUNT)) {
                        // Get particle from shared memory
                        neighbor = &myParticles[i - firstParticleIdx];
                    } else {
                        // Get particle from global memory
                        neighbor = &particles[i];
                    }
                    if (neighbor->cellID != neighborCellIdx)
                        break;
                    particle->density +=
                        MASS * densityKernel(particle, neighbor);
                }
            }
        }
    }

    // Update pressure using new density
    particle->pressure = GAS_CONSTANT * (particle->density - REST_DENSITY);

    // Write my particle back to global memory
    particles[pIdx] = *particle;
}

__global__ void kernelUpdateForces(Particle *particles, int *neighborGrid) {
    int pIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int myChunkIdx = blockIdx.x;
    int totalChunks = gridDim.x;

    int startChunkIdx = max(myChunkIdx - (CHUNK_COUNT / 2), 0);

    if ((myChunkIdx + (CHUNK_COUNT / 2)) >= totalChunks) {
        startChunkIdx = max(0, totalChunks - CHUNK_COUNT);
    }

    int firstParticleIdx = startChunkIdx * MAX_THREADS_PER_BLOCK;

    if (pIdx >= deviceSettings.numParticles) {
        return;
    }

    // Shared array to store the particles related to this block
    __shared__ Particle myParticles[MAX_THREADS_PER_BLOCK * CHUNK_COUNT];

    for (int i = 0; i < CHUNK_COUNT; i++) {
        int particleToLoad = (firstParticleIdx + i * blockDim.x) + threadIdx.x;
        if (particleToLoad >= deviceSettings.numParticles) break;
        myParticles[particleToLoad - firstParticleIdx] = particles[particleToLoad];
    }

    __syncthreads();

    Particle *particle = &myParticles[(myChunkIdx - startChunkIdx) * MAX_THREADS_PER_BLOCK + threadIdx.x];
    int3 cell = getGridCell(particle->position);
    particle->force.x = 0.f;
    particle->force.y = 0.f;
    particle->force.z = 0.f;

    // Update forces based on neighbors
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
                int neighborIdx = neighborGrid[neighborCellIdx];
                if (neighborIdx == -1)
                    continue;
                for (int i = neighborIdx; i < deviceSettings.numParticles;
                     i++) {

                    Particle *neighbor = NULL;

                    if ((i >= firstParticleIdx) &&
                        (i < firstParticleIdx +
                                 MAX_THREADS_PER_BLOCK * CHUNK_COUNT)) {
                        // Get particle from shared memory
                        neighbor = &myParticles[i - firstParticleIdx];
                    } else {
                        // Get particle from global memory
                        neighbor = &particles[i];
                    }

                    if (neighbor->cellID != neighborCellIdx)
                        break;

                    // Calculate pressure force
                    float nd = fmaxf(neighbor->density, 1e-3f);
                    float fPressure =
                        -MASS * (particle->pressure + neighbor->pressure) /
                        (2.f * nd);
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
                    float nv = fmaxf(neighbor->density, 1e-3f);
                    float fViscosity = VISCOSITY * MASS * viscosityKernel(particle, neighbor) / nv;
                    dv.x *= fViscosity;
                    dv.y *= fViscosity;
                    dv.z *= fViscosity;
                    particle->force.x += dv.x;
                    particle->force.y += dv.y;
                    particle->force.z += dv.z;
                }
            }
        }
    }
    particle->force.y += MASS * GRAVITY;

    // Write my particle back to global memory
    particles[pIdx] = *particle;
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
        printf("Bad force/velocity at particle %d: fx=%f, vx=%f\n",
               pIdx, particle->force.x, particle->velocity.x);
    }
    if (!isfinite(particle->force.y) || !isfinite(particle->velocity.y)) {
        printf("Bad force/velocity at particle %d: fy=%f, vy=%f\n",
               pIdx, particle->force.y, particle->velocity.y);
    }
    if (!isfinite(particle->force.z) || !isfinite(particle->velocity.z)) {
        printf("Bad force/velocity at particle %d: fz=%f, vz=%f\n",
               pIdx, particle->force.z, particle->velocity.z);
    }

    particle->velocity.x += timestep * particle->force.x / MASS;
    particle->velocity.y += timestep * particle->force.y / MASS;
    particle->velocity.z += timestep * particle->force.z / MASS;

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
    // if (isnan(particle->position.x) || isnan(particle->position.y) || isnan(particle->position.z)) {
    //     printf("NaN found at particle index %d\n", pIdx);
    // }
    // Write updated positions
    devicePosition[pIdx] = particle->position;
}

// Reset the list heads
__global__ void kernelResetGrid(int *neighborGrid) {
    int listIdx = blockIdx.x + blockIdx.y * gridDim.y +
                  blockIdx.z * gridDim.z * gridDim.z;

    neighborGrid[listIdx] = -1;
}

// Induce velocity on mouse click
__global__ void kernelMoveParticles(Particle *particles, int *neighborGrid,
                                    int2 mouse_pos) {
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
            int neighborIdx = neighborGrid[neighborCellIdx];
            if (neighborIdx == -1)
                continue;
            for (int i = neighborIdx; i < deviceSettings.numParticles; i++) {
                Particle *neighbor = &particles[i];
                if (neighbor->cellID != neighborCellIdx)
                    break;
                if (dx != 0)
                    neighbor->velocity.x += (1.f / dx) * PUSH_STRENGTH;
                if (dy != 0)
                    neighbor->velocity.y += (1.f / dy) * PUSH_STRENGTH;
                if (dx == 0 && dy == 0)
                    neighbor->velocity.z -= PUSH_STRENGTH;
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

    // Free the grid on device
    if (neighborGrid != NULL) {
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
                                  neighborGridDim * sizeof(int));
    cudaMalloc(&particles, settings->numParticles * sizeof(Particle));
    cudaMalloc(&devicePosition, settings->numParticles * sizeof(float3));

    cudaMemset(neighborGrid, -1,
               neighborGridDim * neighborGridDim * neighborGridDim *
                   sizeof(int));
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
        int count = 0;
        for (float x = settings->h; x < settings->boxDim - settings->h;
             x += spacing) {
            for (float y = settings->h; y < settings->boxDim - settings->h;
                 y += spacing) {
                for (float z = settings->h; z < settings->boxDim - settings->h;
                     z += spacing) {
                    tmpParticles[count] = Particle(make_float3(x, y, z));
                    count++;
                    if (count >= settings->numParticles)
                        break;
                }
                if (count >= settings->numParticles)
                    break;
            }
            if (count >= settings->numParticles)
                break;
        }
    }

    // Copy initialized particles to device
    cudaMemcpy(particles, tmpParticles,
               settings->numParticles * sizeof(Particle),
               cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(deviceSettings, settings, sizeof(Settings));
}

void Simulator::buildNeighborGrid() {
    // Assign cell id of each particle
    dim3 blockDim(MAX_THREADS_PER_BLOCK);
    dim3 gridDim((settings->numParticles + MAX_THREADS_PER_BLOCK - 1) /
                 MAX_THREADS_PER_BLOCK);

    kernelAssignCellID<<<gridDim, blockDim>>>(particles);

    // Sort particles array by cell id
    thrust::sort(thrust::device, particles, particles + settings->numParticles,
                 ParticleComp());

    // Populate neighborGrid
    kernelPopulateGrid<<<gridDim, blockDim>>>(particles, neighborGrid);
}

void Simulator::simulate() {
    // Build neighbor grid
    buildNeighborGrid();

    // Compute updates
    dim3 blockDim(MAX_THREADS_PER_BLOCK);
    dim3 gridDim((settings->numParticles + MAX_THREADS_PER_BLOCK - 1) /
                 MAX_THREADS_PER_BLOCK);

    kernelUpdatePressureAndDensity<<<gridDim, blockDim>>>(particles,
                                                          neighborGrid);
    kernelUpdateForces<<<gridDim, blockDim>>>(particles, neighborGrid);
    kernelUpdatePositions<<<gridDim, blockDim>>>(particles, devicePosition);
    cudaDeviceSynchronize();

    // Copy updated positions to host
    cudaMemcpy(position, devicePosition,
               sizeof(float3) * settings->numParticles, cudaMemcpyDeviceToHost);

    // Handle mouse click
    if (mouseClicked) {
        dim3 blockDimClick(settings->numCellsPerDim);
        dim3 gridDimClick(1);

        kernelMoveParticles<<<gridDimClick, blockDimClick>>>(
            particles, neighborGrid, clickCoords);
        cudaDeviceSynchronize();
        mouseClicked = false;
    }
}

void Simulator::simulateAndTime(Times *times) {
    // Build neighbor grid
    auto buildGridStart = std::chrono::steady_clock::now();

    buildNeighborGrid();

    times->buildGrid +=
        std::chrono::duration_cast<std::chrono::duration<double>>(
            std::chrono::steady_clock::now() - buildGridStart)
            .count();

    // Compute updates
    auto sphUpdateStart = std::chrono::steady_clock::now();

    dim3 blockDim(MAX_THREADS_PER_BLOCK);
    dim3 gridDim((settings->numParticles + MAX_THREADS_PER_BLOCK - 1) /
                 MAX_THREADS_PER_BLOCK);

    kernelUpdatePressureAndDensity<<<gridDim, blockDim>>>(particles,
                                                          neighborGrid);
    kernelUpdateForces<<<gridDim, blockDim>>>(particles, neighborGrid);
    kernelUpdatePositions<<<gridDim, blockDim>>>(particles, devicePosition);
    cudaDeviceSynchronize();

    times->sphUpdate +=
        std::chrono::duration_cast<std::chrono::duration<double>>(
            std::chrono::steady_clock::now() - sphUpdateStart)
            .count();

    // Copy updated positions to host
    auto memcpyStart = std::chrono::steady_clock::now();

    cudaMemcpy(position, devicePosition,
               sizeof(float3) * settings->numParticles, cudaMemcpyDeviceToHost);

    times->memcpy += std::chrono::duration_cast<std::chrono::duration<double>>(
                         std::chrono::steady_clock::now() - memcpyStart)
                         .count();
    times->iters += 1;
}
