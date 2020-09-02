
////////////////////////////////////////////////////////////////////////////////
//
//              ****************************
//              *** MC-GPU , version 1.4 ***
//              ****************************
//                                          
//!   Header file containing the declarations for the MC-GPU code.
//!   This file declares all the host and device functions and structures,
//!   the library files to include in the compilation, various constants parameters
//!   of the simulation, pre-processor macro functions, etc.
//
//
//
//          ** DISCLAIMER **
//
// This software and documentation (the "Software") were developed at the Food and
// Drug Administration (FDA) by employees of the Federal Government in the course
// of their official duties. Pursuant to Title 17, Section 105 of the United States
// Code, this work is not subject to copyright protection and is in the public
// domain. Permission is hereby granted, free of charge, to any person obtaining a
// copy of the Software, to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish, distribute,
// sublicense, or sell copies of the Software or derivatives, and to permit persons
// to whom the Software is furnished to do so. FDA assumes no responsibility
// whatsoever for use by other parties of the Software, its source code,
// documentation or compiled executables, and makes no guarantees, expressed or
// implied, about its quality, reliability, or any other characteristic. Further,
// use of this code in no way implies endorsement by the FDA or confers any
// advantage in regulatory decisions.  Although this software can be redistributed
// and/or modified freely, we ask that any derivative works bear some notice that
// they are derived from it, and any modified versions bear some notice that they
// have been modified.
//                                                                            
//
//!                     @file    MC-GPU_v1.5b.h
//!                     @author  Andreu Badal (Andreu.Badal-Soler{at}fda.hhs.gov)
//!                     @date    2018/01/01
//
////////////////////////////////////////////////////////////////////////////////


#ifndef MCGPU_H_
#define MCGPU_H_

// *** To use MPI to simulate multiple CT projections in different GPUs compile with "-DUSING_MPI" or uncomment the following line:
//#define USING_MPI


//! MPI macro: mark commands to be executed only by the main thread (myID==0).
#define MAIN_THREAD if(0==myID)

//! Maximum number of projections allowed in the CT simulation (not limited by the constant memory because stored in global and shared memory):
#define  MAX_NUM_PROJECTIONS  720

//! Maximum number of materials allowed in the input file.
#define  MAX_MATERIALS      15

//! Constants values for the Compton and Rayleigh models:
#define  MAX_SHELLS         30
#define  NP_RAYLEIGH       128
#define  MAX_ENERGYBINS_RAYLEIGH  25005

//! Maximum number of energy bins in the input x-ray energy spectrum.
#define  MAX_ENERGY_BINS   256


#define  PI      3.14159265358979323846
#define  RAD2DEG 180.0/PI
#define  DEG2RAD PI/180.0

// Other constants used in the code:
//! Value to scale the deposited energy in the pixels so that it can be stored as a long long integer
//! instead of a double precision float. The integer values have to be used in order to use the
//! atomicAdd function in CUDA.
#define SCALE_eV        100.0f

//! Offset value to make sure the particles are completely inside, or outside, the voxel bounding box.
//! For example, to start a particle track the source may have to translate a particle from the focal spot to the plane Y=0, but 
//! in reality the particle will be moved to Y=EPS_SOURCE to make sure it is unmistakenly located inside a voxel.
//! If "EPS_SOURCE" is too small an artifact with weird concentric circular shadows may appear at the center of the simulated image.
#define EPS_SOURCE      0.000025f

#define NEG_EPS_SOURCE -EPS_SOURCE
#define INF             500000.0f
#define NEG_INF        -500000.0f

#define EPS             1.5e-6f

//! The largest unsigned int value will mark that a particle escaped the voxelized region.
#define FLAG_OUTSIDE_VOXELS 4294967295

//! Scaling factor for the total number of histories in a mammography projection versus all tomosynthesis projections. Used when flag_simulateMammoAfterDBT is true.
#define SCALE_MAMMO_DBT (2.0/3.0)

//! Preprocessor macro to calculate maximum and minimum values:
#define max_value( a, b ) ( ((a) > (b)) ? (a) : (b) )
#define min_value( a, b ) ( ((a) < (b)) ? (a) : (b) )


// Include standard header files:
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include "zlib.h"    // Library used to read gzip material and voxel files (non-compressed files can also be read). Compile with option -lz
#ifdef USING_MPI
  #include <mpi.h>
#endif

// Include CUDA runtime and helper functions:
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <vector_types.h>


float density_LUT[MAX_MATERIALS];                     // Storing the material densities (nominal or user-input) in a global array in CPU and in constant memory in GPU
int voxelId[256];                                     // Storing the voxel-to-material conversion table in a global array.
__constant__ float density_LUT_CONST[MAX_MATERIALS];  // Density look-up table in GPU constant memory


// - Definition of the vector structures used in the GPU for CPU use:
//   struct int2  { int x, y; };         typedef struct int2 int2;
//   struct int3  { int x, y, z; };      typedef struct int3 int3;
//   struct float2 { float x, y; };      typedef struct float2 float2;
//   struct float3 { float x, y, z; };   typedef struct float3 float3;
//   struct double2 { double x, y; };    typedef struct double2 double2;
//   struct double3 { double x, y, z; }; typedef struct double3 double3;
//   struct short3 { short x, y, z; };   typedef struct short3 short3;  
//   struct ulonglong2 { unsigned long long int x, y; }; typedef struct ulonglong2 ulonglong2;
//   struct uchar3 {unsigned char x, y, z; }; typedef struct uchar3 uchar3;
 


// MC-GPU structure declarations:

//! Structure storing the data defining the source model (except for the energy spectrum).
//! When a CT is simulated, multiple sources will be stored in an array (one source instance per projection angle).
struct __align__(16) 
source_struct       // Define a cone beam x-ray source.
{
  float3 position,            // Input focal spot position at angle 0
         rotation_point,      // Center of rotation for the source in a tomographic acquisition: input source location and direction multiplied by source-detector dist
         axis_of_rotation,    // If tomographic acquisition, axis of rotation for the source (perpendicular to the source-detector vector)
         direction;           // The actual source direction is not really used in the kernel, the rot_fan rotation is applied to the sampled direction around (0,1,0)
  float rot_fan[9],      // Rotation (from axis (0,1,0)) defined by the source direction (needed by the fan beam source; its inverse is used by the detector).
        cos_theta_low,   // Angles for the fan beam model (pyramidal source).
        phi_low,
        D_cos_theta,
        D_phi,
        max_height_at_y1cm,
        max_width_at_y1cm,
        focal_spot_FWHM,
        rotation_blur,
        angle_offset,         // Angular rotation to first projection (input source direction considered as 0 degrees; typically negative offset for DBT)
        angle_per_projection; // Angle increment between projections
  bool flag_halfConeX;        // For a DBT acquisition, block the cone beam towards negative X (ie, beam axis aligned to chest wall)
};


//! Structure storing the data defining the x-ray detector. 
//! For a CT, the struct stores for each angle the detector location and the rotations to
//! transport the detector to a plane perpendicular to +Y.
//! To simulate multiple projections, an array of MAX_NUM_PROJECTIONS of instances of this struct have to be defined.
struct __align__(16) 
detector_struct         // Define a 2D detector plane, located in front of the defined source (centered at the focal spot and perpendicular to the initial direction).
{                       // The radiograohic image will be stored in the global variable "unsigned long long int *image".
  float3 center;
  float2 offset;  // Offset of the image on the detector plane (width, height directions) wrt the default position with beam center at center of image.

  float rot_inv[9],    // Rotation to transport a particle on the detector plane to a frame where the detector is perpendicular to +Y.
        width_X,
        height_Z,
        inv_pixel_size_X,
        inv_pixel_size_Z,
        sdd,                            // Store the source-detector distance
        scintillator_MFP,
        scintillator_thickness,
        kedge_energy,
        fluorescence_yield,
        fluorescence_energy,
        fluorescence_MFP,
        cover_thickness,
        cover_MFP,
        grid_freq,
        grid_ratio,                     // Grid orientation encoded in this variable: <0 --> 0: strips parallel to image width, >0 --> 1: strips parallel to image height
        grid_strip_thickness,
        grid_strip_mu,                  // Coefficient of attenuation for the attenuating strips and the interspace material [1/cm]
        grid_interspace_mu,
        gain_W,                         // Note: the folowing 3 values are not used in the kernel, but still copied to GPU memory
        Swank_rel_std,
        electronic_noise;

  int2 num_pixels;
  int total_num_pixels;       
};


//! Structure storing the source energy spectrum data to be sampled using the Walker aliasing algorithm.
struct __align__(16) 
source_energy_struct       // Define a cone beam x-ray source.
{  
  int num_bins_espc;                     // Number of bins in the input energy spectrum
  float espc[MAX_ENERGY_BINS],           // Energy values of the input x-ray energy spectrum
        espc_cutoff[MAX_ENERGY_BINS];    // Cutoffs for the Walker aliasing sampling of the spectrum
  short int espc_alias[MAX_ENERGY_BINS]; // Aliases for the Walker aliasing sampling algorithm (stored as short to save memory)
};


//! Structure defining a voxelized box with the back-lower corner at the coordinate origin.
struct __align__(16) 
voxel_struct                     // Define a voxelized box with the back-lower corner at the coordinate origin.
{                                // Voxel material and densities are stored in a local variable.
  int3 num_voxels;
  float3 inv_voxel_size,
         voxel_size,
         size_bbox,
         offset,                 // Offset of voxel geometry (default origin at lower back corner)
         voxel_size_HiRes;
  uchar3 num_voxels_coarse;      // Store the size of the coarse voxels that will be described by a binary tree
};


//! Structure with the basic data required by the linear interpolation of the mean free paths: number of values and energy grid.
struct __align__(16) 
linear_interp       // Constant data for linear interpolation of mean free paths
{                                        // The parameters 'a' and 'b' are stored in local variables float4 *a, *b;
  int num_values;      // -->  Number of iterpolation points (eg, 2^12 = 4096).
  float e0,            // --> Minimum energy
        ide;           // --> Inverse of the energy bin width
};


//! Structure storing the data of the Compton interaction sampling model (equivalent to PENELOPE's common block /CGCO/).
struct __align__(16) 
compton_struct      // Data from PENELOPE's common block CGCO: Compton interaction data
{
  float fco[MAX_MATERIALS*MAX_SHELLS],
        uico[MAX_MATERIALS*MAX_SHELLS],
        fj0[MAX_MATERIALS*MAX_SHELLS];
  int noscco[MAX_MATERIALS];
};

//! Structure storing the data of the Rayleigh interaction sampling model (equivalent to PENELOPE's common block /CGRA/).
struct __align__(16) 
rayleigh_struct
{
  float xco[NP_RAYLEIGH*MAX_MATERIALS],
        pco[NP_RAYLEIGH*MAX_MATERIALS],
        aco[NP_RAYLEIGH*MAX_MATERIALS],
        bco[NP_RAYLEIGH*MAX_MATERIALS],
        pmax[MAX_ENERGYBINS_RAYLEIGH*MAX_MATERIALS];
  unsigned char itlco[NP_RAYLEIGH*MAX_MATERIALS],
                ituco[NP_RAYLEIGH*MAX_MATERIALS];
};



//// *** HOST FUNCTIONS *** ////

void read_input(int argc, char** argv, int myID, unsigned long long int* total_histories, int* gpu_id, int* seed_input, int* num_threads_per_block, int* histories_per_thread, struct detector_struct* detector_data, unsigned long long int** image_ptr, int* image_bytes, struct source_struct* source_data, struct source_energy_struct* source_energy_data, struct voxel_struct* voxel_data, char* file_name_voxels, char file_name_materials[MAX_MATERIALS][250], char* file_name_output, char* file_name_espc, int* num_projections, ulonglong2** voxels_Edep_ptr, int* voxels_Edep_bytes, char* file_dose_output, short int* dose_ROI_x_min, short int* dose_ROI_x_max, short int* dose_ROI_y_min, short int* dose_ROI_y_max, short int* dose_ROI_z_min, short int* dose_ROI_z_max, double* SRotAxisD, double* translation_helical, int* flag_material_dose, bool* flag_simulateMammoAfterDBT, bool* flag_detectorFixed);
void load_voxels(int myID, char* file_name_voxels, float* density_max, struct voxel_struct* voxel_data, int** voxel_mat_dens_ptr, long long int* voxel_mat_dens_bytes, short int* dose_ROI_x_max, short int* dose_ROI_y_max, short int* dose_ROI_z_max);
void load_material(int myID, char file_name_materials[MAX_MATERIALS][250], float* density_max, float* density_nominal, struct linear_interp* mfp_table_data, float2** mfp_Woodcock_table, int* mfp_Woodcock_table_bytes, float3** mfp_table_a_ptr, float3** mfp_table_b_ptr, int* mfp_table_bytes, struct rayleigh_struct *rayleigh_table_ptr, struct compton_struct *compton_table_ptr);
void trim_name(char* input_line, char* file_name);
char* fgets_trimmed(char* trimmed_line, int num, FILE* file_ptr);
int report_image(char* file_name_output, struct detector_struct* detector_data, struct source_struct* source_data, float mean_energy_spectrum, unsigned long long int* image, double time_elapsed, unsigned long long int total_histories, int current_projection, int num_projections, int myID, int numprocs, double current_angle, int* seed_input);
void set_CT_trajectory(int myID, int num_projections, struct source_struct* source_data, struct detector_struct* detector_data, double translation_helical, bool flag_detectorFixed);
int report_voxels_dose(char* file_dose_output, int num_projections, struct voxel_struct* voxel_data, int* voxel_mat_dens, ulonglong2* voxels_Edep, double time_elapsed_total, unsigned long long int total_histories, short int dose_ROI_x_min, short int dose_ROI_x_max, short int dose_ROI_y_min, short int dose_ROI_y_max, short int dose_ROI_z_min, short int dose_ROI_z_max, struct source_struct* source_data);
void init_energy_spectrum(char* file_name_espc, struct source_energy_struct* source_energy_data, float *mean_energy_spectrum);
void update_seed_PRNG(int batch_number, unsigned long long int total_histories, int* seed);
void IRND0(float *W, float *F, short int *K, int N);
int report_materials_dose(int num_projections, unsigned long long int total_histories, float *density_nominal, ulonglong2 *materials_dose, double *mass_materials, char file_name_materials[MAX_MATERIALS][250]);


int seeki_walker(float *cutoff, short int *alias, float randno, int n);   // (This function is not actually called in the code.)


int guestimate_GPU_performance(int gpu_id);
void init_CUDA_device( int* gpu_id, int myID, int numprocs,
      /*Variables to GPU constant memory:*/ struct voxel_struct* voxel_data, struct source_struct* source_data, struct source_energy_struct* source_energy_data, struct detector_struct* detector_data, struct linear_interp* mfp_table_data,
      /*Variables to GPU global memory:*/ int* voxel_mat_dens, int** voxel_mat_dens_device, long long int voxel_mat_dens_bytes,
        char* bitree, char** bitree_device, unsigned int bitree_bytes,
        unsigned long long int* image, unsigned long long int** image_device, int image_bytes,
        float2* mfp_Woodcock_table, float2** mfp_Woodcock_table_device, int mfp_Woodcock_table_bytes,
        float3* mfp_table_a, float3* mfp_table_b, float3** mfp_table_a_device, float3** mfp_table_b_device, int mfp_table_bytes,
        struct rayleigh_struct* rayleigh_table, struct rayleigh_struct** rayleigh_table_device,
        struct compton_struct* compton_table, struct compton_struct** compton_table_device,
        struct detector_struct** detector_data_device, struct source_struct** source_data_device,
        ulonglong2* voxels_Edep, ulonglong2** voxels_Edep_device, int voxels_Edep_bytes, short int* dose_ROI_x_min, short int* dose_ROI_x_max, short int* dose_ROI_y_min, short int* dose_ROI_y_max, short int* dose_ROI_z_min, short int* dose_ROI_z_max,
        ulonglong2* materials_dose, ulonglong2** materials_dose_device, int flag_material_dose, int** seed_input_device, int* seed_input, int num_projections);



//// *** DEVICE CONSTANT MEMORY DECLARATION (~global variables in the GPU) *** ////

// -- Constant memory (defined as global variables):

//! Global variable to be stored in the GPU constant memory defining the coordinates of the dose deposition region of interest.
__constant__ short int dose_ROI_x_min_CONST, dose_ROI_x_max_CONST, dose_ROI_y_min_CONST, dose_ROI_y_max_CONST, dose_ROI_z_min_CONST, dose_ROI_z_max_CONST;

//! Global variable to be stored in the GPU constant memory defining the size of the voxel phantom.
__constant__ struct voxel_struct    voxel_data_CONST;      // Define the geometric constants

//! Global variable to be stored in the GPU constant memory defining the linear interpolation data.
__constant__ struct linear_interp   mfp_table_data_CONST;  // Define size of interpolation arrays

//! Global variable to be stored in the GPU constant memory defining the source energy spectrum.
__constant__ struct source_energy_struct source_energy_data_CONST;


//// *** GLOBAL FUNCTIONS *** ////

__global__ void init_image_array_GPU(unsigned long long int* image, int pixels_per_image);
__global__ void init_dose_array_GPU(ulonglong2* voxels_Edep, int num_voxels_dose);
__global__ void track_particles(int histories_per_thread, short int num_p, int* seed_input_device, unsigned long long int* image, ulonglong2* voxels_Edep, int* voxel_mat_dens, char* bitree, float2* mfp_Woodcock_table, float3* mfp_table_a, float3* mfp_table_b, struct rayleigh_struct* rayleigh_table, struct compton_struct* compton_table, struct detector_struct* detector_data_array, struct source_struct* source_data_array, ulonglong2* materials_dose);


//// *** DEVICE FUNCTIONS *** ////

__device__ inline void tally_image(float* energy, float3* position, float3* direction, signed char* scatter_state, unsigned long long int* image, struct source_struct* source_data_SHARED, struct detector_struct* detector_data_SHARED, int2* seed);
__device__ inline void source(float3* position, float3* direction, float* energy, int2* seed, unsigned int* absvox, struct source_struct* source_data_SHARED, struct detector_struct* detector_data_SHARED);
__device__ inline void move_to_bbox(float3* position, float3* direction, unsigned int* intersection_flag);
__device__ inline void init_PRNG(int history_batch, int histories_per_thread, int seed_input, int2* seed);
__host__ __device__ inline int abMODm(int m, int a, int s);
__device__ inline float ranecu(int2* seed);
__device__ inline double ranecu_double(int2* seed);
__host__ inline double ranecu_double_CPU(int2* seed);
__device__ inline unsigned int locate_voxel(float3 position, short3* voxel_coord);
__device__ inline void rotate_double(float3* direction, double cos_theta, double phi);
__device__ inline void GRAa(float *energy, double *costh_Rayleigh, int *mat, float *pmax_current, int2 *seed, struct rayleigh_struct* cgra);
__device__ inline void GCOa(float *energy, double *costh_Compton, int *mat, int2 *seed, struct compton_struct* cgco_SHARED);
__device__ inline void tally_voxel_energy_deposition(float* Edep, short3* voxel_coord, ulonglong2* dose);
__device__ inline void tally_materials_dose(float* Edep, int* material, ulonglong2* materials_dose);

__device__ inline float sample_gausspdf_below2sigma(int2 *seed);     // Return Gaussian distributed random value, with distribution cropped at 2 sigma.
inline void gausspdf_double_CPU(double *g1, double *g2, int2 *seed); // Return two Gaussian distributed random values (as doubles).
__device__ inline void gausspdf(float *g1, float *g2, int2 *seed);   // Return two Gaussian distributed random values (as floats).

__device__ __host__ inline void rotate_around_axis_Rodrigues(float *angle, float3 *w, float3 *p);  // Rotate vector p around vector w the input angle using Rodrigues' formula: http://mathworld.wolfram.com/RodriguesRotationFormula.html
__device__ __host__ inline void rotate_2vectors_around_axis_Rodrigues(float *angle, float3 *w, float3 *p, float3 *v);
__device__ __host__ inline void apply_rotation(float3 *v, float *m);

void create_rotation_matrix_around_axis(float angle, float wx, float wy, float wz, float *m);
inline void multiply_3x3(float *m_out, float *m, float *n);

__device__ float antiscatter_grid_transmission_prob(float3* position, float3* direction, struct detector_struct* detector_data);
__device__ int find_material_bitree(const float3* position, char* bitree, const int bitree_root_index, short3* voxel_coord);


// -- END OF THE "ifndef MCGPU_H_" statement:
#endif


