# CUDA SPH Simulation

## Overview

This repository contains the source code for 3 different implementations of a smoothed-particle hydrodynamics (SPH) fluid simulation using CUDA. The implementations vary in the data structure used for the neighbor search, and the source code for each can be found in the corresponding sub-directory of this project. In particular, the `main` branch code uses an array of lock-free linked lists to represent the neighbor lists, the `index_sort` branch uses an array of particle indices sorted by flattened grid index, and the `z_index_sort` branch uses an array of particle indices sorted by the Morton-encoded grid index.

This simulation was written in completion of the final project for [15-618](https://www.cs.cmu.edu/~418/schedule.html) at CMU. See our [final report](https://drive.google.com/file/d/1sxU1pIbQUkyNjprhUSueufVPywgfyecm/view) for a comprehensive description of the code and its performance.

## Usage
Before running the code, first ensure you are logged in to one of the GHC machines (47-86) with an NVIDIA RTX 2080 GPU. If you intend to visualize the simulation and are accessing the machine remotely, further ensure you have enabled X forwarding. Finally, ensure NVCC is added to your path. See page 2 of https://www.cs.cmu.edu/afs/cs/academic/class/15418-s25/public/asst/asst2/asst2_handout.pdf for more details.

Inside one of the subdirectories of this project, run `make` to generate the executable. The simulation can then be launched using the following command:

```
./sph -n <NUM_PARTICLES> -i <random/grid> -m <free/time>
```

where the `-n` argument specifies the number of particles simulated, the `-i` argument corresponds to the particle position initialization strategy, and the `-m` mode corresponds to the execution mode. To visualize the simulation, use `-m free`, and to only report timing information for 100 timesteps, use `-m time`. In `free` mode, clicking near the particles will cause the fluid to ripple slightly.