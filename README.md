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

