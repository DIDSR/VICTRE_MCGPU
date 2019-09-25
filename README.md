# MC-GPU_v1.5b: VICTRE pivotal study simulations

This version of MC-GPU was developed exclusively to replicate as realistically as possible a Siemens Mammomat Inspiration system for the Virtual Imaging Clinical Trial for Regulatory Evaluation (VICTRE) project.
This code was used in the VICTRE pivotal study simulations executed in February 2018.

The software is designed to simulate full-field digital mammography images and digital breast tomosynthesis (DBT) scans of the computational breast phantoms created by Christian Graff's software (https://github.com/DIDSR/BreastPhantom).
Some of the improvements implemented in this version of MC-GPU compared to previous versions (https://github.com/DIDSR/MCGPU) are:
- DBT acquisition geometry.
- Extended focal spot model.
- Optional focal spot motion.
- Optional anti-scatter grid model.
- Amorphous Selenium direct detector model: depth of interaction, fluorescence escape, charge generation, Swank factor, electronic noise.
- Voxelized phantoms stored in memory usig a binary tree structure to save memory.
- *Limitation: material densities hardcoded for Graff's phantom composition.*

## [Example simulation](example_simulations/README.md)
Simulation input files, and auxiliary files, have been added in the [example_simulations folder](https://github.com/DIDSR/VICTRE_MCGPU/tree/master/example_simulations) to allow the replication of two of the simulations of the VICTRE project.
A breast phantom with scattered glandularity and a heterogreously dense phantom created with C. Graff model are provided (these phantoms were not part of the original VICTRE pivotal study, and have a single large mass embedded inside). The material files (generated from PENELOPE 2006 material files) and energy spectra used in the simulations are included.

<img src="example_simulations/results/mcgpu_image_22183101_scattered_0000_a.jpg" width="900px"/>

## Disclaimer

This software and documentation (the "Software") were developed at the Food and Drug Administration (FDA) by employees of the Federal Government in the course of their official duties. Pursuant to Title 17, Section 105 of the United States Code, this work is not subject to copyright protection and is in the public domain. Permission is hereby granted, free of charge, to any person obtaining a copy of the Software, to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, or sell copies of the Software or derivatives, and to permit persons to whom the Software is furnished to do so. FDA assumes no responsibility whatsoever for use by other parties of the Software, its source code, documentation or compiled executables, and makes no guarantees, expressed or implied, about its quality, reliability, or any other characteristic. Further, use of this code in no way implies endorsement by the FDA or confers any advantage in regulatory decisions. Although this software can be redistributed and/or modified freely, we ask that any derivative works bear some notice that they are derived from it, and any modified versions bear some notice that they have been modified.

