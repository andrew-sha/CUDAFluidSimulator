#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <iostream>

#include "simulator.h"

#define MAX_THREADS_PER_BLOCK (1024)

__constant__ Settings deviceSettings;

// GPU helper functions
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
  }
}

const float *Simulator::getPosition() {
  std::cout << "Copying position data from device" << std::endl;

  // cudaMemcpy(position,
  //            cudaDevicePosition,
  //            sizeof(float) * 3 * settings->numParticles,
  //            cudaMemcpyDeviceToHost);

  return position;
}

void Simulator::setup() {
  // create a position array for the device which will be copied back to the
  // host

  int neighborGridDim = settings->boxDim / settings->h;
  cudaMalloc(&neighborGrid, neighborGridDim * neighborGridDim *
                                neighborGridDim * sizeof(Particle *));
  cudaMalloc(&particles, settings->numParticles * sizeof(Particle));

  Particle *cpuParticles =
      (Particle *)malloc(sizeof(Particle) * settings->numParticles);

  if (position == NULL) {
    position = (float *)malloc(sizeof(float) * 3 * settings->numParticles);
  }

  // set random initial particle positions
  for (size_t i = 0; i < settings->numParticles; i++) {
    float x = rand() / (float)RAND_MAX * settings->boxDim;
    float y = rand() / (float)RAND_MAX * settings->boxDim;
    float z = rand() / (float)RAND_MAX * settings->boxDim;

    cpuParticles[i] = Particle(make_float3(x, y, z));
  }

  // Copy the porticles to device
  cudaMemcpy(particles, cpuParticles, settings->numParticles * sizeof(Particle),
             cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(deviceSettings, settings, sizeof(Settings));
}

void Simulator::simulate() {
  // 1. Build the grid
  dim3 blockDim(MAX_THREADS_PER_BLOCK);
  dim3 gridDim((settings->numParticles + MAX_THREADS_PER_BLOCK - 1) /
               MAX_THREADS_PER_BLOCK);

  kernelBuildGrid<<<gridDim, blockDim>>>(particles, neighborGrid);

  // 2. Compute updates
  // 3. Send positions back to host
  // 4. Reset the heads of the linked lists

  return;
}
