## Example simulation

Input files for four example simulations with two breast phantoms, one with scattered glandularity (scattered) and the other heterogeneously dense (hetero) are provided. 
These phantoms were created with [C. Graff model](https://github.com/DIDSR/breastPhantom), and have a large mass embedded inside.
Two of the input files (named "fast"), intended to get results quickly (around 5 minutes), simulate a mammogram of the phantom with 1% of the number x-rays in a regular examination. 
The other two input files (named "mammo+DBT") simulate, in a single execution, a full dose mammogram and a DBT scan with 25 individual projections (all projections together use 50% more x rays than the mammogram alone).
Note that the simulations with the scattered phantom will take substantially longer due to the larger thickness and increased field of view (more of the detector being irradiated, requiring more x-ray tracks).

The material files (generated from PENELOPE 2006 material files), energy spectra files, phantom data, and sample results and output files are provided in the sub-folders.
A compiled executable file (Ubuntu 16.04.6 LTS, CUDA-9.0) and example results and output files simulated in a NVIDIA GeForce GTX TITAN X card are provided in the results folder.

To compile the source code, run the simulation, and visualize the out image, execute:

```bash
sh make_MC-GPU_v1.5b.sh
time ./MC-GPU_v1.5b.x MC-GPU_v1.5b_scattered_phantom_mammo_fast.in | tee MC-GPU_v1.5b_scattered_phantom_mammo_fast.out 
imagej results/mcgpu_image_22183101_scattered_mammo.raw &
```
The output images are stored in raw format with one floating point value per pixel. They can be imported into imagej using: Image type: 32-bit real; width: 3000 pixels; height: 1500 pixels; Number of Images: 1 or 2 (fist image shows scatter and primary radiation, the second shows primary x rays only); white is zero; Little-endian byte order.

If multiple GPUs are available in the current computer, the simulation will be able to use them in parallel using MPI communications (if the GPUs are in different computers, add the options "-x LD_LIBRARY_PATH -hostfile my_hostfile.txt"). To execute the simulation in parallel in 4 GPUs, use the command:
```bash
mpirun -n 4 /path/MC-GPU_v1.5b.x /path/MC-GPU_v1.5b_scattered_phantom_mammo_fast.in | tee /path/MC-GPU_v1.5b_scattered_phantom_mammo_fast.out
```

