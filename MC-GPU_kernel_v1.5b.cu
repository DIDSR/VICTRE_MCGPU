
////////////////////////////////////////////////////////////////////////////////
//
//              ****************************
//              *** MC-GPU, version 1.5b ***
//              ****************************
//                                          
//!  Definition of the CUDA GPU kernel for the simulation of x ray tracks in a voxelized geometry.
//!  The physics models for Rayleigh and Compton scattering are translated from the Fortran
//!  code in PENELOPE 2006.
//
//        ** DISCLAIMER **
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
//!                     @file    MC-GPU_kernel_v1.5b.cu
//!                     @author  Andreu Badal (Andreu.Badal-Soler{at}fda.hhs.gov)
//!                     @date    2018/01/01
//                       -- Original code started on:  2009/04/14
//
////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
//!  Initialize the image array, ie, set all pixels to zero
//!  Essentially, this function has the same effect as the command: 
//!   "cutilSafeCall(cudaMemcpy(image_device, image, image_bytes, cudaMemcpyHostToDevice))";
//!  
//!  CUDA performs some initialization work the first time a GPU kernel is called.
//!  Therefore, calling a short kernel before the real particle tracking is performed
//!  may improve the accuracy of the timing measurements in the relevant kernel.
//!  
//!       @param[in,out] image   Pointer to the image array.
//!       @param[in] pixels_per_image  Number of pixels in the image (ie, elements in the array).
////////////////////////////////////////////////////////////////////////////////
__global__ void init_image_array_GPU(unsigned long long int* image, int pixels_per_image)
{
  int my_pixel = threadIdx.x + blockIdx.x*blockDim.x;
  if (my_pixel < pixels_per_image)
  {
    // -- Set the current pixel to 0 and return, avoiding overflow when more threads than pixels are used:
    image[my_pixel] = (unsigned long long int)(0);    // Initialize non-scatter image
    my_pixel += pixels_per_image;                     //  (advance to next image)
    image[my_pixel] = (unsigned long long int)(0);    // Initialize Compton image
    my_pixel += pixels_per_image;                     //  (advance to next image)
    image[my_pixel] = (unsigned long long int)(0);    // Initialize Rayleigh image
    my_pixel += pixels_per_image;                     //  (advance to next image)
    image[my_pixel] = (unsigned long long int)(0);    // Initialize multi-scatter image
  }
}

// ////////////////////////////////////////////////////////////////////////////////
// //!  Initialize the dose deposition array, ie, set all voxel doses to zero
// //!  
// //!       @param[in,out] dose   Pointer to the dose mean and sigma arrays.
// //!       @param[in] num_voxels_dose  Number of voxels in the dose ROI (ie, elements in the arrays).
// ////////////////////////////////////////////////////////////////////////////////
// __global__
// void init_dose_array_GPU(ulonglong2* voxels_Edep, int num_voxels_dose)
// {  
//   int my_voxel = threadIdx.x + blockIdx.x*blockDim.x;
//   register ulonglong2 ulonglong2_zero;
//   ulonglong2_zero.x = ulonglong2_zero.y = (unsigned long long int) 0;
//   if (my_voxel < num_voxels_dose)
//   {
//     dose[my_voxel] = ulonglong2_zero;    // Set the current voxel to (0,0) and return, avoiding overflow
//   }
// }


 
////////////////////////////////////////////////////////////////////////////////
//!  Main function to simulate x-ray tracks inside a voxelized geometry.
//!  Secondary electrons are not simulated (in photoelectric and Compton 
//!  events the energy is locally deposited).
//!
//!  The following global variables, in  the GPU __constant__ memory are used:
//!           voxel_data_CONST, 
//!           source_energy_data_CONST
//!           mfp_table_data_CONST.
//!           density_LUT_CONST
//!
//!       @param[in] history_batch  Particle batch number (only used in the CPU version when CUDA is disabled!, the GPU uses the built-in variable threadIdx)
//!       @param[in] num_p  Projection number in the CT simulation. This variable defines a specific angle and the corresponding source and detector will be used.
//!       @param[in] histories_per_thread   Number of histories to simulate for each call to this function (ie, for GPU thread).
//!       @param[in] seed_input   Random number generator seed (the same seed is used to initialize the two MLCGs of RANECU).
//!       @param[in] voxel_mat_dens   Pointer to the voxel densities and material vector (the voxelized geometry), stored in GPU glbal memory.
//!       @param[in] mfp_Woodcock_table    Two parameter table for the linear interpolation of the Woodcock mean free path (MFP) (stored in GPU global memory).
//!       @param[in] mfp_table_a   First element for the linear interpolation of the interaction mean free paths (stored in GPU global memory).
//!       @param[in] mfp_table_b   Second element for the linear interpolation of the interaction mean free paths (stored in GPU global memory).
//!       @param[in] rayleigh_table   Pointer to the table with the data required by the Rayleigh interaction sampling, stored in GPU global memory.
//!       @param[in] compton_table   Pointer to the table with the data required by the Compton interaction sampling, stored in GPU global memory.
//!       @param[in,out] image   Pointer to the image vector in the GPU global memory.
//!       @param[in,out] dose   Pointer to the array containing the 3D voxel dose (and its uncertainty) in the GPU global memory.
////////////////////////////////////////////////////////////////////////////////
__global__ void track_particles(int histories_per_thread,
                                short int num_p,                       // For a CT simulation: allocate space for up to MAX_NUM_PROJECTIONS projections.
                                int* seed_input_device,                // Random seed read from global memory; secuence continued for successive projections in same GPU.   !!DBTv1.4!!
                                unsigned long long int* image,
                                ulonglong2* voxels_Edep,
                                int* voxel_mat_dens,                   //!!bitree!! Using "int" to be store the index to the bitree table       //!!FixedDensity_DBT!! Allocating "voxel_mat_dens" as "char" instead of "float2"
                                char* bitree,                          //!!bitree!! Array with the bitrees for every non-uniform coarse voxel
                                float2* mfp_Woodcock_table,
                                float3* mfp_table_a,
                                float3* mfp_table_b,
                                struct rayleigh_struct* rayleigh_table,
                                struct compton_struct* compton_table,
                                struct detector_struct* detector_data_array,
                                struct source_struct* source_data_array, 
                                ulonglong2* materials_dose)
{
  // -- Declare the track state variables:
  float3 position, direction;
  float energy, step, prob, randno, mfp_density, mfp_Woodcock;
  float3 mfp_table_read_a, mfp_table_read_b;
  int2 seed;
  int index;
  int material0,        // Current material, starting at 0 for 1st material
      material_old;     // Flag to mark a material or energy change
  signed char scatter_state;    // Flag for scatter images: scatter_state=0 for non-scattered, =1 for Compton, =2 for Rayleigh, and =3 for multiple scatter.

  // -- Store the Compton table in shared memory from global memory:
  //    For Compton and Rayleigh the access to memory is not coherent and the caching capability do not speeds up the accesses, they actually slows down the acces to other data.
  __shared__  struct compton_struct cgco_SHARED;
  __shared__  struct detector_struct detector_data_SHARED;
  __shared__  struct source_struct source_data_SHARED;

      if (0==threadIdx.x)  // First GPU thread copies the variables to shared memory
  {

    // -Copy the current source, detector data from global to shared memory for fast access:
    source_data_SHARED    = source_data_array[num_p];
    detector_data_SHARED  = detector_data_array[num_p];    // Copy the long array to a single instance in shared memory for the current projection

    // -Copy the compton data to shared memory:
    cgco_SHARED = *compton_table;
    
  }
  __syncthreads();     // Make sure all threads will see the initialized shared variable  

  // -- Initialize the RANECU generator in a position far away from the previous history:
  init_PRNG((threadIdx.x + blockIdx.x*blockDim.x), histories_per_thread, *seed_input_device, &seed);   // Using a 1D block. Random seed read from global memory.  !!DBTv1.4!!
  
  // -- Loop for the "histories_per_thread" particles in the current history_batch:

  for( ; histories_per_thread>0; histories_per_thread--)
  {
        //  printf("\n\n********* NEW HISTORY:  %d    [seeds: %d, %d]\n\n", histories_per_thread, seed.x, seed.y); //  fflush(stdout);  // !!Verbose!! calling printf from the GPU is possible but if multiple threads call it at the same time some output will be lost.

    unsigned int absvox = 1;

    // -- Call the source function to get a primary x ray:
    source(&position, &direction, &energy, &seed, &absvox, &source_data_SHARED, &detector_data_SHARED);
    
    scatter_state = (signed char)0;     // Reset previous scatter state: new non-scattered particle loaded

    // -- Find the current energy bin by truncation (this could be pre-calculated for a monoenergetic beam):    
    //    The initialization host code made sure that the sampled energy will always be within the tabulated energies (index never negative or too large).
    index = __float2int_rd((energy-mfp_table_data_CONST.e0)*mfp_table_data_CONST.ide);  // Using CUDA function to convert float to integer rounding down (towards minus infinite)

  
    // -- Get the minimum mfp at the current energy using linear interpolation (Woodcock tracking):      
    {
      float2 mfp_Woodcock_read = mfp_Woodcock_table[index];   // Read the 2 parameters for the linear interpolation in a single read from global memory
      mfp_Woodcock = mfp_Woodcock_read.x + energy * mfp_Woodcock_read.y;   // Interpolated minimum MFP          
    }


    // -- Reset previous material to force a recalculation of the MFPs (negative materials are not allowed in the voxels):
    material_old  = -1;

    // *** X-ray interaction loop:
    for(;;)
    {
      
      if (absvox==FLAG_OUTSIDE_VOXELS)
          break;    // -- Primary particle was not pointing to the voxel region! (but may still be detected after moving in vacuum in a straight line).      


      // *** Virtual interaction loop:  // New loop structure in MC-GPU_v1.3: simulate all virtual events before sampling Compton & Rayleigh: 
      
//       float2 matdens;
      short3 voxel_coord;    // Variable used only by DOSE TALLY

      do
      {     

        step = -(mfp_Woodcock)*logf(ranecu(&seed));   // Using the minimum MFP in the geometry for the input energy (Woodcock tracking)

        position.x += step*direction.x;
        position.y += step*direction.y;
        position.z += step*direction.z;

 
        // -- Locate the new particle in the voxel geometry:      
        absvox = locate_voxel(position, &voxel_coord);   // Get the voxel number at the current position and the voxel coordinates (used to check if inside the dose ROI in DOSE TALLY).
        if (absvox==FLAG_OUTSIDE_VOXELS)
          break;    // -- Particle escaped the voxel region! ("index" is still >0 at this moment)
          
              //         matdens = voxel_mat_dens[absvox];     // Get the voxel material and density in a single read from global memory
              //         material0 = (int)(matdens.x - 1);   // Set the current material by truncation, and set 1st material to value '0'.

        //!!FixedDensity_DBT!! Allocating "voxel_mat_dens" as "char" instead of "float2". Density taken from function "density_LUT". First material number == 0
        material0 = (int)voxel_mat_dens[absvox];     // Get the voxel material and density in a single read from global memory (first material==0) 

        if (material0<0)
        {
          // -- Non-uniform low resolution voxel: find material at current location searching the original high resolution geometry using the corresponding binary tree:
          material0 = find_material_bitree(&position, bitree, -material0, &voxel_coord);    // !!bitree!!         
        }
        
        // -- Get the data for the linear interpolation of the interaction MFPs, in case the energy or material have changed:
        if (material0 != material_old)
        {
          mfp_table_read_a = mfp_table_a[index*(MAX_MATERIALS)+material0];
          mfp_table_read_b = mfp_table_b[index*(MAX_MATERIALS)+material0];
          material_old = material0;                                              // Store the new material
        }
        
        // *** Apply Woodcock tracking:

        mfp_density = mfp_Woodcock * density_LUT_CONST[material0];      //!!FixedDensity_DBT!! Density taken from constant memory array "density_LUT_CONST"; Old: mfp_density=mfp_Woodcock*matdens.y;
        
        // -- Calculate probability of delta scattering, using the total mean free path for the current material and energy (linear interpolation):
        prob = 1.0f - mfp_density * (mfp_table_read_a.x + energy * mfp_table_read_b.x);
        randno = ranecu(&seed);    // Sample uniform PRN
      }
      while (randno<prob);   // [Iterate if there is a delta scattering event]

      if (absvox==FLAG_OUTSIDE_VOXELS)
        break;    // -- Particle escaped the voxel region! Break the interaction loop to call tally image.

        
      // The GPU threads will be stopped and waiting here until ALL threads have a REAL event: 

      // -- Real event takes place! Check the kind of event and sample the effects of the interaction:
      
      prob += mfp_density * (mfp_table_read_a.y + energy * mfp_table_read_b.y);    // Interpolate total Compton MFP ('y' component)
      if (randno<prob)   // [Checking Compton scattering]
      {
        // *** Compton interaction:

        //  -- Sample new direction and energy:
        double costh_Compton;
        randno = energy;     // Save temporal copy of the particle energy (variable randno not necessary until next sampling). DOSE TALLY
        
        GCOa(&energy, &costh_Compton, &material0, &seed, &cgco_SHARED);
        rotate_double(&direction, costh_Compton, /*phi=2*pi*PRN=*/ 6.28318530717958647693*ranecu_double(&seed));

        randno = energy - randno;   // Save temporal copy of the negative of the energy lost in the interaction.  DOSE TALLY

        // -- Find the new energy interval:
        index = __float2int_rd((energy-mfp_table_data_CONST.e0)*mfp_table_data_CONST.ide);  // Using CUDA function to convert float to integer rounding down (towards minus infinite)
        
        if (index>-1)  // 'index' will be negative only when the energy is below the tabulated minimum energy: particle will be then absorbed (rejected) after tallying the dose.
        {          
          // -- Get the Woodcock MFP for the new energy (energy above minimum cutoff):
          float2 mfp_Woodcock_read = mfp_Woodcock_table[index];   // Read the 2 parameters for the linear interpolation in a single read from global memory
          mfp_Woodcock = mfp_Woodcock_read.x + energy * mfp_Woodcock_read.y;   // Interpolated minimum MFP

          material_old = -2;    // Set an impossible material to force an update of the MFPs data for the nex energy interval

          // -- Update scatter state:
          if (scatter_state==(signed char)0)
            scatter_state = (signed char)1;   // Set scatter_state == 1: Compton scattered particle
          else
            scatter_state = (signed char)3;   // Set scatter_state == 3: Multi-scattered particle
        }

      }
      else
      {
        prob += mfp_density * (mfp_table_read_a.z + energy * mfp_table_read_b.z);    // Interpolate total Rayleigh MFP ('z' component)
        if (randno<prob)   // [Checking Rayleigh scattering]
        {
          // *** Rayleigh interaction:

          //  -- Sample angular deflection:
          double costh_Rayleigh;
          float pmax_current = rayleigh_table->pmax[(index+1)*MAX_MATERIALS+material0];   // Get max (ie, value for next bin?) cumul prob square form factor for Rayleigh sampling

          GRAa(&energy, &costh_Rayleigh, &material0, &pmax_current, &seed, rayleigh_table);
          rotate_double(&direction, costh_Rayleigh, /*phi=2*pi*PRN=*/ 6.28318530717958647693*ranecu_double(&seed));

          // -- Update scatter state:
          if (scatter_state==(signed char)0)
            scatter_state = (signed char)2;   // Set scatter_state == 1: Rayleigh scattered particle
          else
            scatter_state = (signed char)3;   // Set scatter_state == 3: Multi-scattered particle

        }
        else
        {
          // *** Photoelectric interaction (or pair production): mark particle for absorption after dose tally (ie, index<0)!
          randno = -energy;   // Save temporal copy of the (negative) energy deposited in the interaction (variable randno not necessary anymore).
          index = -11;       // A negative "index" marks that the particle was absorved and that it will never arrive at the detector.
        }
      }
    
      //  -- Tally the dose deposited in Compton and photoelectric interactions:
      if (randno<-0.001f)
      {
        float Edep = -1.0f*randno;   // If any energy was deposited, this variable will temporarily store the negative value of Edep.
        
        //  -- Tally the dose deposited in the current material, if enabled (ie, array allocated and not null):
        if (materials_dose!=NULL)
          tally_materials_dose(&Edep, &material0, materials_dose);    // !!tally_materials_dose!!

        //  -- Tally the energy deposited in the current voxel, if enabled (tally disabled when dose_ROI_x_max_CONST is negative). DOSE TALLY
          
            // Optional code to skip dose tally in air (material=0):  if (dose_ROI_x_max_CONST > -1 && 0!=material0)
        if (dose_ROI_x_max_CONST > -1)
          tally_voxel_energy_deposition(&Edep, &voxel_coord, voxels_Edep);

      }    

      // -- Break interaction loop for particles that have been absorbed or with energy below the tabulated cutoff: particle is "absorbed" (ie, track discontinued).
      if (index<0)
        break;  
      
    }   // [Cycle the X-ray interaction loop]

    if (index>-1)
    {
      // -- Particle escaped the voxels but was not absorbed, check if it will arrive at the detector and tally its energy:      
      tally_image(&energy, &position, &direction, &scatter_state, image, &source_data_SHARED, &detector_data_SHARED, &seed);

    }
  }   // [Continue with a new history]

  

  // -- Store the final random seed used by the last thread in the grid to global memory in order to continue the random secuence in successive projections in same GPU without overlapping.                   !!DBTv1.4!!
  //    Since I am only storing the 'x' component and using it to init both parts of the ranecu generator, the sequence will actually diverge, but I warranty that at least one MLCG will stay uncorrelated.   !!DeBuG!!
  if ( (blockIdx.x == (gridDim.x-1)) && (threadIdx.x == (blockDim.x-1)))
  {
    *seed_input_device = seed.x;    // Store seed in GPU memory, but only for the thread with the largest id
  }

  
  
}   // [All tracks simulated for this kernel call: return to CPU]






////////////////////////////////////////////////////////////////////////////////
//!  Tally the dose deposited in the voxels.
//!  This function is called whenever a particle suffers a Compton or photoelectric
//!  interaction. It is not necessary to call this function if the dose tally
//!  was disabled in the input file (ie, dose_ROI_x_max_CONST < 0).
//!  Electrons are not transported in MC-GPU and therefore we are approximating
//!  that the dose is equal to the KERMA (energy released by the photons alone).
//!  This approximation is acceptable when there is electronic equilibrium and when
//!  the range of the secondary electrons is shorter than the voxel size. Usually the
//!  doses will be acceptable for photon energies below 1 MeV. The dose estimates may
//!  not be accurate at the interface of low density volumes.
//!
//!  We need to use atomicAdd() in the GPU to prevent that multiple threads update the 
//!  same voxel at the same time, which would result in a lose of information.
//!  This is very improbable when using a large number of voxels but gives troubles 
//!  with a simple geometries with few voxels (in this case the atomicAdd will slow 
//!  down the code because threads will update the voxel dose secuentially).
//!
//!
//!       @param[in] Edep   Energy deposited in the interaction
//!       @param[in] voxel_coord   Voxel coordinates, needed to check if particle located inside the input region of interest (ROI)
//!       @param[out] voxels_Edep   ulonglong2 array containing the 3D voxel dose and dose^2 (ie, uncertainty) as unsigned integers scaled by SCALE_eV.
////////////////////////////////////////////////////////////////////////////////
__device__ inline void tally_voxel_energy_deposition(float* Edep, short3* voxel_coord, ulonglong2* voxels_Edep)
{ 
  if((voxel_coord->x < dose_ROI_x_min_CONST) || (voxel_coord->x > dose_ROI_x_max_CONST) ||
     (voxel_coord->y < dose_ROI_y_min_CONST) || (voxel_coord->y > dose_ROI_y_max_CONST) ||
     (voxel_coord->z < dose_ROI_z_min_CONST) || (voxel_coord->z > dose_ROI_z_max_CONST))
    {
      return;   // -- Particle outside the ROI: return without tallying anything.
    }

  // -- Particle inside the ROI: tally Edep.
  register int DX = 1 + (int)(dose_ROI_x_max_CONST - dose_ROI_x_min_CONST);
  register int num_voxel = (int)(voxel_coord->x-dose_ROI_x_min_CONST) + ((int)(voxel_coord->y-dose_ROI_y_min_CONST))*DX + ((int)(voxel_coord->z-dose_ROI_z_min_CONST))*DX*(1 + (int)(dose_ROI_y_max_CONST-dose_ROI_y_min_CONST));
  
  atomicAdd(&voxels_Edep[num_voxel].x, __float2ull_rn((*Edep)*SCALE_eV) );    // Energy deposited at the voxel, scaled by the factor SCALE_eV and rounded.
  atomicAdd(&voxels_Edep[num_voxel].y, __float2ull_rn((*Edep)*(*Edep)) );     // (not using SCALE_eV for std_dev to prevent overflow)           

  return;
}



////////////////////////////////////////////////////////////////////////////////
//!  Source that creates primary x rays, according to the defined source model.
//!  The particles are automatically moved to the surface of the voxel bounding box,
//!  to start the tracking inside a real material. If the sampled particle do not
//!  enter the voxels, it is init in the focal spot and the main program will check
//!  if it arrives at the detector or not.
//!
//!       @param[in] source_data   Structure describing the source.
//!       @param[in] source_energy_data_CONST   Global variable in constant memory space describing the source energy spectrum.
//!       @param[out] position   Initial particle position (particle transported inside the voxel bbox).
//!       @param[out] direction   Sampled particle direction (cosine vectors).
//!       @param[out] energy   Sampled energy of the new x ray.
//!       @param[in] seed   Current seed of the random number generator, requiered to sample the movement direction.
//!       @param[out] absvox   Set to <0 if primary particle will not cross the voxels, not changed otherwise (>0).
////////////////////////////////////////////////////////////////////////////////
__device__ inline void source(float3* position, float3* direction, float* energy, int2* seed, unsigned int* absvox, struct source_struct* source_data_SHARED, struct detector_struct* detector_data_SHARED)
{
  // *** Sample the initial x-ray energy following the input energy spectrum using the Walker aliasing algorithm from PENELOPE:
      // The following code is equivalent to calling the function "seeki_walker": int sampled_bin = seeki_walker(source_data_CONST.espc_cutoff, source_data_CONST.espc_alias, ranecu(seed), source_data_CONST.num_bins_espc);      
  int sampled_bin;
  float RN = ranecu(seed) * source_energy_data_CONST.num_bins_espc;    // Find initial interval (array starting at 0):   
  int int_part = __float2int_rd(RN);                          //   -- Integer part (round down)
  float fraction_part = RN - ((float)int_part);                 //   -- Fractional part
  if (fraction_part < source_energy_data_CONST.espc_cutoff[int_part])  // Check if we are in the aliased part
    sampled_bin = int_part;                                     // Below the cutoff: return current value
  else
    sampled_bin = (int)source_energy_data_CONST.espc_alias[int_part];  // Above the cutoff: return alias
  
  // Linear interpolation of the final energy within the sampled energy bin:
  *energy = source_energy_data_CONST.espc[sampled_bin] + ranecu(seed) * (source_energy_data_CONST.espc[sampled_bin+1] - source_energy_data_CONST.espc[sampled_bin]);   



  // *** If not a point source, sample the focal spot position using a uniformly-distributed angle on a sphere AND a Gaussian-distributed random radius:       !!DBTv1.4!!  
  if (source_data_SHARED->focal_spot_FWHM > 5.0e-7f)
  {
    float g = sample_gausspdf_below2sigma(seed);   // Return a Gaussian distributed random value located at less than 2 sigma from the center.          !!DBTv1.4!!
      // Cropping the Gaussian dist at 2 sigma to prevent generating photons unrealistically far from the focal spot center. The 2 sigma limit has been set arbitrary and will affect 4.55% of sampled locations.  
      // Experimental focal spot measurements show that the spot is quite sharp [A Burgess, "Focal spots: I. MTF separability", Invest Radiol 12, p. 36-43 (1977)]
    
        //ALTERNATIVE METHOD:     float g = sample_gausspdf(seed);   // Return a Gaussian distributed random value.            !!DBTv1.4!!
        //ALTERNATIVE METHOD:     gausspdf(&g1, &g2, seed);   // Sample 2 independent Gaussian distributed random variables.

    float cos_thetaFS = 2.0f*ranecu(seed)-1.0f;   // Sample uniform points on a sphere
    float sin_thetaFS = sqrtf(1.0f-cos_thetaFS*cos_thetaFS);
    float phiFS = (PI*2.0f)*ranecu(seed);    
    float cos_phiFS, sin_phiFS;
    sincos(phiFS, &sin_phiFS, &cos_phiFS);    
      // Full Width at Half Maximum for Gaussian curve:  FWHM  =  [2*sqrt(2*ln(2))] * sigma  =  2.3548 * sigma
      // For a focal spot with FWHM = 0.0200 cm --> sigma = 0.0200/2.354820 = 0.0200*0.4246609 = 0.008493
    float r = g * source_data_SHARED->focal_spot_FWHM * 0.424660900144f;     // Use a Gaussian distribution for the radius
    
    // Set current focal spot position with sampled focal spot shift (source_data_SHARED->position was already rotated to the appropriate angle):
    position->x = source_data_SHARED->position.x + r*sin_thetaFS*cos_phiFS;
    position->y = source_data_SHARED->position.y + r*sin_thetaFS*sin_phiFS;
    position->z = source_data_SHARED->position.z + r*cos_thetaFS;
  }
  else
  {
    // Set default focal spot position for point source:
    position->x = source_data_SHARED->position.x;
    position->y = source_data_SHARED->position.y;
    position->z = source_data_SHARED->position.z;
  }


  // *** Sample the initial direction:
   
  do   //  Iterate sampling if the sampled direction is not acceptable to get a square field at the given phi (rejection sampling): force square field for any phi!!
  {
    //     Using the algorithm used in PENMAIN.f, from penelope 2008 (by F. Salvat).
    direction->z = source_data_SHARED->cos_theta_low + ranecu(seed)*source_data_SHARED->D_cos_theta;     // direction->z = w = cos(theta_sampled)
    register float phi_sampled = source_data_SHARED->phi_low + ranecu(seed)*source_data_SHARED->D_phi;
    register float sin_theta_sampled = sqrtf(1.0f - direction->z*direction->z);
    float sinphi_sampled, cosphi_sampled;
    
    sincos(phi_sampled, &sinphi_sampled,&cosphi_sampled);    // Calculate the SIN and COS at the same time.    
    direction->y = sin_theta_sampled * sinphi_sampled;
    direction->x = sin_theta_sampled * cosphi_sampled;
  }
  while( (fabsf(direction->z/(direction->y+1.0e-8f)) > source_data_SHARED->max_height_at_y1cm) ||    // Force square field for any phi by rejection sampling. (The "+1e-8" prevents division by zero)
         (fabsf(direction->x/(direction->y+1.0e-8f)) > source_data_SHARED->max_width_at_y1cm) );     //!!DBTv1.4!!


  // -- Apply the rotation that moves the emission direction from the default direction pointing to (0,1,0), to the required acquistion orientation:
  apply_rotation(direction, source_data_SHARED->rot_fan);    //!!DBTv1.4!!
  
  
  // *** Simulate motion blur (if needed): Rotate focal spot position and emission direction according to a uniformly-sampled angular motion blur    !!DBTv1.4!!
  if (source_data_SHARED->rotation_blur>EPS)
  {
    position->x -= source_data_SHARED->rotation_point.x;    // Move to the coordinate system where rotation point is at the origin to apply the rotation
    position->y -= source_data_SHARED->rotation_point.y;
    position->z -= source_data_SHARED->rotation_point.z;
  
    float blur_angle = source_data_SHARED->rotation_blur*(ranecu(seed)-0.5f);    // Uniform sampling of angular motion blur before and after the nominal acquisition angle
    
//     rotate_around_axis_Rodrigues(&blur_angle, &source_data_SHARED->axis_of_rotation, position);  // Rotate position around rotation angle using Rodrigues' formula (http://mathworld.wolfram.com/RodriguesRotationFormula.html)
    rotate_2vectors_around_axis_Rodrigues(&blur_angle, &source_data_SHARED->axis_of_rotation, position, direction);  // Rotate position and direction around rotation angle using Rodrigues' formula (http://mathworld.wolfram.com/RodriguesRotationFormula.html)

    position->x += source_data_SHARED->rotation_point.x;    // Move back to the real-world coordinate system where rotation point is not at the origin
    position->y += source_data_SHARED->rotation_point.y;
    position->z += source_data_SHARED->rotation_point.z;
  }

  
  // To be safe, renormalize the direction vector to 1 (should not be necessary but single precision math might accumulate errors)
  double NORM = rsqrt(direction->x*direction->x + direction->y*direction->y + direction->z*direction->z);     // !!DeBuG!! Check if it is really necessary to renormalize in a real simulation!!
  direction->x = NORM*direction->x;
  direction->y = NORM*direction->y;
  direction->z = NORM*direction->z;
       //        printf("%.20lf   %.20lf   %.20lf\n", NORM, rsqrt(direction->x*direction->x + direction->y*direction->y + direction->z*direction->z), diff);   //!!VERBOSE!!  !!DeBuG!!


  // *** Move the particle to the inside of the voxel bounding box:
  move_to_bbox(position, direction, absvox);
}



////////////////////////////////////////////////////////////////////////////////
//!  Functions to moves a particle towards the inside of the voxelized geometry bounding box.
//!  An EPSILON distance is added to make sure the particles will be clearly inside the bbox, 
//!  not exactly on the surface. 
//!
//!  This algorithm makes the following assumptions:
//!     - The back lower vertex of the voxel bounding box is always located at the origin: (x0,y0,z0)=(0,0,0).
//!     - The initial value of "position" corresponds to the focal spot location.
//!     - When a ray is not pointing towards the bbox plane that it should cross according to the sign of the direction,
//!       I assign a distance to the intersection =0 instead of the real negative distance. The wall that will be 
//!       crossed to enter the bbox is always the furthest and therefore a 0 distance will never be used except
//!       in the case of a ray starting inside the bbox or outside the bbox and not pointing to any of the 3 planes. 
//!       In this situation the ray will be transported a 0 distance, meaning that it will stay at the focal spot.
//!
//!  (Interesting information on ray-box intersection: http://tog.acm.org/resources/GraphicsGems/gems/RayBox.c)
//!
//!       @param[in,out] position Particle position: initially set to the focal spot, returned transported inside the voxel bbox.
//!       @param[out] direction   Sampled particle direction (cosine vectors).
//!       @param[out] intersection_flag   Set to <0 if particle outside bbox and will not cross the voxels, not changed otherwise.
//!       @param[in] size_bbox    Global variable from structure voxel_data_CONST: size of the bounding box.
//!       @param[in] offset       Global variable from structure voxel_data_CONST: offset of the geometry in x, y, and z.
////////////////////////////////////////////////////////////////////////////////
__device__ inline void move_to_bbox(float3* position, float3* direction, unsigned int* intersection_flag)
{
  float dist_y, dist_x, dist_z;

  // -Distance to the nearest Y plane:
  if ((direction->y) > EPS_SOURCE)   // Moving to +Y: check distance to y=0 plane
  {
    // Check Y=0 (bbox wall):
    if (position->y > voxel_data_CONST.offset.y)     //!!DBTv1.4!! Allowing a 3D offset of the voxelized geometry (default origin at lower back corner).
      dist_y = 0.0f;  // No intersection with this plane: particle inside or past the box  
          // The actual distance would be negative but we set it to 0 bc we will not move the particle if no intersection exist.
    else
      dist_y = EPS_SOURCE + (voxel_data_CONST.offset.y-position->y)/(direction->y);    // dist_y > 0 for sure in this case
  }
  else if ((direction->y) < NEG_EPS_SOURCE)
  {
    // Check Y=voxel_data_CONST.size_bbox.y:
    if (position->y < (voxel_data_CONST.size_bbox.y + voxel_data_CONST.offset.y))
      dist_y = 0.0f;  // No intersection with this plane
    else
      dist_y = EPS_SOURCE + (voxel_data_CONST.size_bbox.y + voxel_data_CONST.offset.y - position->y)/(direction->y);    // dist_y > 0 for sure in this case
  }
  else   // (direction->y)~0
    dist_y = NEG_INF;   // Particle moving parallel to the plane: no interaction possible (set impossible negative dist = -INFINITE)

  // -Distance to the nearest X plane:
  if ((direction->x) > EPS_SOURCE)
  {
    // Check X=0:
    if (position->x > voxel_data_CONST.offset.x)
      dist_x = 0.0f;
    else  
      dist_x = EPS_SOURCE + (voxel_data_CONST.offset.x-position->x)/(direction->x);    // dist_x > 0 for sure in this case
  }
  else if ((direction->x) < NEG_EPS_SOURCE)
  {
    // Check X=voxel_data_CONST.size_bbox.x:
    if (position->x < (voxel_data_CONST.size_bbox.x+voxel_data_CONST.offset.x))
      dist_x = 0.0f;
    else  
      dist_x = EPS_SOURCE + (voxel_data_CONST.size_bbox.x + voxel_data_CONST.offset.x - position->x)/(direction->x);    // dist_x > 0 for sure in this case
  }
  else
    dist_x = NEG_INF;

  // -Distance to the nearest Z plane:
  if ((direction->z) > EPS_SOURCE)
  {
    // Check Z=0:
    if (position->z > voxel_data_CONST.offset.z)
      dist_z = 0.0f;
    else
      dist_z = EPS_SOURCE + (voxel_data_CONST.offset.z - position->z)/(direction->z);    // dist_z > 0 for sure in this case
  }
  else if ((direction->z) < NEG_EPS_SOURCE)
  {
    // Check Z=voxel_data_CONST.size_bbox.z:
    if (position->z < (voxel_data_CONST.size_bbox.z+voxel_data_CONST.offset.z))
      dist_z = 0.0f;
    else
      dist_z = EPS_SOURCE + (voxel_data_CONST.size_bbox.z + voxel_data_CONST.offset.z - position->z)/(direction->z);    // dist_z > 0 for sure in this case
  }
  else
    dist_z = NEG_INF;

  
  // -- Find the longest distance plane, which is the one that has to be crossed to enter the bbox.
  //    Storing the maximum distance in variable "dist_z". Distance will be =0 if no intersection exists or 
  //    if the x ray is already inside the bbox.
  if ( (dist_y>dist_x) && (dist_y>dist_z) )
    dist_z = dist_y;      // dist_z == dist_max 
  else if (dist_x>dist_z)
    dist_z = dist_x;
// else
//   dist_max = dist_z;
    
  // -- Move particle from the focal spot (current location) to the bbox wall surface (slightly inside):
  float x = position->x + dist_z * direction->x;
  float y = position->y + dist_z * direction->y;
  float z = position->z + dist_z * direction->z;      
  
  // Check if the new position is outside the bbox. If not, return the moved location:
  if ( (x < voxel_data_CONST.offset.x) || (x > (voxel_data_CONST.size_bbox.x+voxel_data_CONST.offset.x)) || 
       (y < voxel_data_CONST.offset.y) || (y > (voxel_data_CONST.size_bbox.y+voxel_data_CONST.offset.y)) || 
       (z < voxel_data_CONST.offset.z) || (z > (voxel_data_CONST.size_bbox.z+voxel_data_CONST.offset.z)) )
  {
    (*intersection_flag) = FLAG_OUTSIDE_VOXELS;   // OLD: -111;  // Particle outside the bbox AND not pointing to the bbox: set absvox<0 to skip interaction sampling. Leave particle position at focal spot.
  }
  else
  {
    position->x = x;
    position->y = y;
    position->z = z;
  }

}


////////////////////////////////////////////////////////////////////////////////


//!  Upper limit of the number of random values sampled in a single track.
//!  I need a large leap for simulations containing a heavy element that causes a lot of delta scattering (eg, for a 15 keV simulation with bone and water I might have 10 delta scatterings; adding Tungsten I might have >650 deltas, and each delta iteration consumes two PRN).
#define  LEAP_DISTANCE    2048
      // #define  LEAP_DISTANCE     256     //!!DeBuG!! !!DBTv1.4!! 256 is too low when using Tungsten!!! 
//!  Multipliers and moduli for the two MLCG in RANECU.
#define  a1_RANECU       40014
#define  m1_RANECU  2147483563
#define  a2_RANECU       40692
#define  m2_RANECU  2147483399
////////////////////////////////////////////////////////////////////////////////
//! Initialize the pseudo-random number generator (PRNG) RANECU to a position
//! far away from the previous history (leap frog technique).
//!
//! Each calculated seed initiates a consecutive and disjoint sequence of
//! pseudo-random numbers with length LEAP_DISTANCE, that can be used to
//! in a parallel simulation (Sequence Splitting parallelization method).
//! The basic equation behind the algorithm is:
//!    S(i+j) = (a**j * S(i)) MOD m = [(a**j MOD m)*S(i)] MOD m  ,
//! which is described in:
//!   P L'Ecuyer, Commun. ACM 31 (1988) p.742
//!
//! This function has been adapted from "seedsMLCG.f", see:
//!   A Badal and J Sempau, Computer Physics Communications 175 (2006) p. 440-450
//!
//!       @param[in] history   Particle bach number.
//!       @param[in] seed_input   Initial PRNG seed input (used to initiate both MLCGs in RANECU).
//!       @param[out] seed   Initial PRNG seeds for the present history.
//!
////////////////////////////////////////////////////////////////////////////////
__device__ inline void init_PRNG(int history_batch, int histories_per_thread, int seed_input, int2* seed)
{
  // -- Move the RANECU generator to a unique position for the current batch of histories:
  //    I have to use an "unsigned long long int" value to represent all the simulated histories in all previous batches
  //    The maximum unsigned long long int value is ~1.8e19: if history >1.8e16 and LEAP_DISTANCE==1000, 'leap' will overflow.
  // **** 1st MLCG:
  unsigned long long int leap = ((unsigned long long int)(history_batch+1))*(histories_per_thread*LEAP_DISTANCE);
  int y = 1;
  int z = a1_RANECU;
  // -- Calculate the modulo power '(a^leap)MOD(m)' using a divide-and-conquer algorithm adapted to modulo arithmetic
  for(;;)
  {
    // (A2) Halve n, and store the integer part and the residue
    if (0!=(leap&01))  // (bit-wise operation for MOD(leap,2), or leap%2 ==> proceed if leap is an odd number)  Equivalent: t=(short)(leap%2);
    {
      leap >>= 1;     // Halve n moving the bits 1 position right. Equivalent to:  leap=(leap/2);  
      y = abMODm(m1_RANECU,z,y);      // (A3) Multiply y by z:  y = [z*y] MOD m
      if (0==leap) break;         // (A4) leap==0? ==> finish
    }
    else           // (leap is even)
    {
      leap>>= 1;     // Halve leap moving the bits 1 position right. Equivalent to:  leap=(leap/2);
    }
    z = abMODm(m1_RANECU,z,z);        // (A5) Square z:  z = [z*z] MOD m
  }
  // AjMODm1 = y;                 // Exponentiation finished:  AjMODm = expMOD = y = a^j

  // -- Compute and display the seeds S(i+j), from the present seed S(i), using the previously calculated value of (a^j)MOD(m):
  //         S(i+j) = [(a**j MOD m)*S(i)] MOD m
  //         S_i = abMODm(m,S_i,AjMODm)
  seed->x = abMODm(m1_RANECU, seed_input, y);     // Using the input seed as the starting seed

  // **** 2nd MLCG (repeating the previous calculation for the 2nd MLCG parameters):
  leap = ((unsigned long long int)(history_batch+1))*(histories_per_thread*LEAP_DISTANCE);
  y = 1;
  z = a2_RANECU;
  for(;;)
  {
    // (A2) Halve n, and store the integer part and the residue
    if (0!=(leap&01))  // (bit-wise operation for MOD(leap,2), or leap%2 ==> proceed if leap is an odd number)  Equivalent: t=(short)(leap%2);
    {
      leap >>= 1;     // Halve n moving the bits 1 position right. Equivalent to:  leap=(leap/2);
      y = abMODm(m2_RANECU,z,y);      // (A3) Multiply y by z:  y = [z*y] MOD m
      if (0==leap) break;         // (A4) leap==0? ==> finish
    }
    else           // (leap is even)
    {
      leap>>= 1;     // Halve leap moving the bits 1 position right. Equivalent to:  leap=(leap/2);
    }
    z = abMODm(m2_RANECU,z,z);        // (A5) Square z:  z = [z*z] MOD m
  }
  // AjMODm2 = y;
  seed->y = abMODm(m2_RANECU, seed_input, y);     // Using the input seed as the starting seed
}



/////////////////////////////////////////////////////////////////////
//!  Calculate "(a1*a2) MOD m" with 32-bit integers and avoiding
//!  the possible overflow, using the Russian Peasant approach
//!  modulo m and the approximate factoring method, as described
//!  in:  L'Ecuyer and Cote, ACM Trans. Math. Soft. 17 (1991).
//!
//!  This function has been adapted from "seedsMLCG.f", see: 
//!  Badal and Sempau, Computer Physics Communications 175 (2006)
//!
//!       @param[in] m,a,s  MLCG parameters
//!       @return   (a1*a2) MOD m   
//
//    Input:          0 < a1 < m                                  
//                    0 < a2 < m                                  
//
//    Return value:  (a1*a2) MOD m                                
//
/////////////////////////////////////////////////////////////////////
__device__ __host__ inline int abMODm(int m, int a, int s)
{
  // CAUTION: the input parameters are modified in the function but should not be returned to the calling function! (pass by value!)
  int q, k;
  int p = -m;            // p is always negative to avoid overflow when adding

  // ** Apply the Russian peasant method until "a =< 32768":
  while (a>32768)        // We assume '32' bit integers (4 bytes): 2^(('32'-2)/2) = 32768
  {
    if (0!=(a&1))        // Store 's' when 'a' is odd     Equivalent code:   if (1==(a%2))
    {
      p += s;
      if (p>0) p -= m;
    }
    a >>= 1;             // Half a (move bits 1 position right)   Equivalent code: a = a/2;
    s = (s-m) + s;       // Double s (MOD m)
    if (s<0) s += m;     // (s is always positive)
  }

  // ** Employ the approximate factoring method (a is small enough to avoid overflow):
  q = (int) m / a;
  k = (int) s / q;
  s = a*(s-k*q)-k*(m-q*a);
  while (s<0)
    s += m;

  // ** Compute the final result:
  p += s;
  if (p<0) p += m;

  return p;
}



////////////////////////////////////////////////////////////////////////////////
//! Pseudo-random number generator (PRNG) RANECU returning a float value
//! (single precision version).
//!
//!       @param[in,out] seed   PRNG seed (seed kept in the calling function and updated here).
//!       @return   PRN double value in the open interval (0,1)
//!
////////////////////////////////////////////////////////////////////////////////
__device__ inline float ranecu(int2* seed)
{
  int i1 = (int)(seed->x/53668);
  seed->x = 40014*(seed->x-i1*53668)-i1*12211;

  int i2 = (int)(seed->y/52774);
  seed->y = 40692*(seed->y-i2*52774)-i2*3791;

  if (seed->x < 0) seed->x += 2147483563;
  if (seed->y < 0) seed->y += 2147483399;

  i2 = seed->x-seed->y;
  if (i2 < 1) i2 += 2147483562;

  return (__int2float_rn(i2)*4.65661305739e-10f);   // 4.65661305739e-10 == 1/2147483563
}


////////////////////////////////////////////////////////////////////////////////
//! Pseudo-random number generator (PRNG) RANECU returning a double value.
////////////////////////////////////////////////////////////////////////////////
__device__ inline double ranecu_double(int2* seed)
{
  int i1 = (int)(seed->x/53668);
  seed->x = 40014*(seed->x-i1*53668)-i1*12211;

  int i2 = (int)(seed->y/52774);
  seed->y = 40692*(seed->y-i2*52774)-i2*3791;

  if (seed->x < 0) seed->x += 2147483563;
  if (seed->y < 0) seed->y += 2147483399;

  i2 = seed->x-seed->y;
  if (i2 < 1) i2 += 2147483562;

  return (__int2double_rn(i2)*4.6566130573917692e-10);
}


////////////////////////////////////////////////////////////////////////////////
__host__ inline double ranecu_double_CPU(int2* seed)
{
  int i1 = (int)(seed->x/53668);
  seed->x = 40014*(seed->x-i1*53668)-i1*12211;

  int i2 = (int)(seed->y/52774);
  seed->y = 40692*(seed->y-i2*52774)-i2*3791;

  if (seed->x < 0) seed->x += 2147483563;
  if (seed->y < 0) seed->y += 2147483399;

  i2 = seed->x-seed->y;
  if (i2 < 1) i2 += 2147483562;

  return ((double)(i2)*4.6566130573917692e-10);
}


////////////////////////////////////////////////////////////////////////////////
//! Find the voxel that contains the current position.
//! Report the voxel absolute index and the x,y,z indices.
//! The structure containing the voxel number and size is read from CONSTANT memory.
//!
//!       @param[in] position   Particle position
//!       @param[out] voxel_coord   Pointer to three integer values (short3*) that will store the x,y and z voxel indices.
//!       @return   Returns "absvox", the voxel number where the particle is
//!                 located (negative if position outside the voxel bbox).
//!
////////////////////////////////////////////////////////////////////////////////
__device__ inline unsigned int locate_voxel(float3 p, short3* voxel_coord)
{

  p.x -= voxel_data_CONST.offset.x;    // Translate the coordinate system to a reference where the voxel's lower back corner is at the origin
  p.y -= voxel_data_CONST.offset.y;
  p.z -= voxel_data_CONST.offset.z;
  
    if ( (p.y < EPS) || (p.y > (voxel_data_CONST.size_bbox.y-EPS)) ||
         (p.x < EPS) || (p.x > (voxel_data_CONST.size_bbox.x-EPS)) ||
         (p.z < EPS) || (p.z > (voxel_data_CONST.size_bbox.z-EPS)) )
  {
    // -- Particle escaped the voxelized geometry:
     return FLAG_OUTSIDE_VOXELS;      // OLD CODE:  return -1;       !!DBTv1.4!!
  }
 
  // -- Particle inside the voxelized geometry, find current voxel:
  //    The truncation from float to integer could give troubles for negative coordinates but this will never happen thanks to the IF at the begining of this function.
  //    (no need to use the CUDA function to convert float to integer rounding down (towards minus infinite): __float2int_rd)
  
  register int voxel_coord_x, voxel_coord_y, voxel_coord_z;
  voxel_coord_x = __float2int_rd(p.x * voxel_data_CONST.inv_voxel_size.x);  
  voxel_coord_y = __float2int_rd(p.y * voxel_data_CONST.inv_voxel_size.y);
  voxel_coord_z = __float2int_rd(p.z * voxel_data_CONST.inv_voxel_size.z);  
  
  voxel_coord->x = (short int) voxel_coord_x;  // Output the voxel coordinates as short int (2 bytes) instead of int (4 bytes) to save registers; avoid type castings in the calculation of the return value.
  voxel_coord->y = (short int) voxel_coord_y;
  voxel_coord->z = (short int) voxel_coord_z;
  
  return ((unsigned int)(voxel_coord_x + voxel_coord_y*(voxel_data_CONST.num_voxels.x)) + ((unsigned int)voxel_coord_z)*(voxel_data_CONST.num_voxels.x)*(voxel_data_CONST.num_voxels.y));
}



//////////////////////////////////////////////////////////////////////
//!   Rotates a vector; the rotation is specified by giving
//!   the polar and azimuthal angles in the "self-frame", as
//!   determined by the vector to be rotated.
//!   This function is a literal translation from Fortran to C of
//!   PENELOPE (v. 2006) subroutine "DIRECT".
//!
//!    @param[in,out]  (u,v,w)  input vector (=d) in the lab. frame; returns the rotated vector components in the lab. frame
//!    @param[in]  costh  cos(theta), angle between d before and after turn
//!    @param[in]  phi  azimuthal angle (rad) turned by d in its self-frame
//
//    Output:
//      (u,v,w) -> rotated vector components in the lab. frame
//
//    Comments:
//      -> (u,v,w) should have norm=1 on input; if not, it is
//         renormalized on output, provided norm>0.
//      -> The algorithm is based on considering the turned vector
//         d' expressed in the self-frame S',
//           d' = (sin(th)cos(ph), sin(th)sin(ph), cos(th))
//         and then apply a change of frame from S' to the lab
//         frame. S' is defined as having its z' axis coincident
//         with d, its y' axis perpendicular to z and z' and its
//         x' axis equal to y'*z'. The matrix of the change is then
//                   / uv/rho    -v/rho    u \
//          S ->lab: | vw/rho     u/rho    v |  , rho=(u^2+v^2)^0.5
//                   \ -rho       0        w /
//      -> When rho=0 (w=1 or -1) z and z' are parallel and the y'
//         axis cannot be defined in this way. Instead y' is set to
//         y and therefore either x'=x (if w=1) or x'=-x (w=-1)
//////////////////////////////////////////////////////////////////////
__device__ inline void rotate_double(float3* direction, double costh, double phi)   // The direction vector is single precision but the rotation is performed in double precision for increased accuracy.
{
  double DXY, NORM, cosphi, sinphi, SDT;
  DXY = direction->x*direction->x + direction->y*direction->y;
  sincos(phi, &sinphi,&cosphi);   // Calculate the SIN and COS at the same time.   sinphi = sin(phi); cosphi = cos(phi);

  // ****  Ensure normalisation
  NORM = DXY + direction->z*direction->z;     // !!DeBuG!! Check if it is really necessary to renormalize in a real simulation!!
  if (fabs(NORM-1.0)>1.0e-14)
  {
    NORM = rsqrt(NORM);
    direction->x = NORM*direction->x;
    direction->y = NORM*direction->y;
    direction->z = NORM*direction->z;
    DXY = direction->x*direction->x + direction->y*direction->y;
  }
  if (DXY>1.0e-28)
  {
    SDT = sqrt((1.0-costh*costh)/DXY);
    float direction_x_in = direction->x;
    direction->x = direction->x*costh + SDT*(direction_x_in*direction->z*cosphi-direction->y*sinphi);
    direction->y = direction->y*costh+SDT*(direction->y*direction->z*cosphi+direction_x_in*sinphi);
    direction->z = direction->z*costh-DXY*SDT*cosphi;
  }
  else
  {
    SDT = sqrt(1.0-costh*costh);
    direction->y = SDT*sinphi;
    if (direction->z>0.0)
    {
      direction->x = SDT*cosphi;
      direction->z = costh;
    }
    else
    {
      direction->x =-SDT*cosphi;
      direction->z =-costh;
    }
  }
}


//////////////////////////////////////////////////////////////////////


//  ***********************************************************************
//  *   Translation of PENELOPE's "SUBROUTINE GRAa" from FORTRAN77 to C   *
//  ***********************************************************************
//!  Sample a Rayleigh interaction using the sampling algorithm
//!  used in PENELOPE 2006.
//!
//!       @param[in] energy   Particle energy (not modified with Rayleigh)
//!       @param[out] costh_Rayleigh   Cosine of the angular deflection
//!       @param[in] material  Current voxel material
//
//  CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
//  C  PENELOPE/PENGEOM (version 2006)                                     C
//  C    Copyright (c) 2001-2006                                           C
//  C    Universitat de Barcelona                                          C
//  C  Permission to use, copy, modify, distribute and sell this software  C
//  C  and its documentation for any purpose is hereby granted without     C
//  C  fee, provided that the above copyright notice appears in all        C
//  C  copies and that both that copyright notice and this permission      C
//  C  notice appear in all supporting documentation. The Universitat de   C
//  C  Barcelona makes no representations about the suitability of this    C
//  C  software for any purpose. It is provided "as is" without express    C
//  C  or implied warranty.                                                C
//  CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
//////////////////////////////////////////////////////////////////////
__device__ inline void GRAa(float *energy, double *costh_Rayleigh, int *mat, float *pmax_current, int2 *seed, struct rayleigh_struct* cgra)
{
/*  ****  Energy grid and interpolation constants for the current energy. */
    double  xmax = ((double)*energy) * 8.065535669099010e-5;       // 8.065535669099010e-5 == 2.0*20.6074/510998.918
    double x2max = min_value( (xmax*xmax) , ((double)cgra->xco[(*mat+1)*NP_RAYLEIGH - 1]) );   // Get the last tabulated value of xco for this mat
    
    if (xmax < 0.01)
    {
       do
       {
          *costh_Rayleigh = 1.0 - ranecu_double(seed) * 2.0;
       }
       while ( ranecu_double(seed) > (((*costh_Rayleigh)*(*costh_Rayleigh)+1.0)*0.5) );
       return;
    }

    for(;;)    // (Loop will iterate everytime the sampled value is rejected or above maximum)
    {
      double ru = ranecu_double(seed) * (double)(*pmax_current);    // Pmax for the current energy is entered as a parameter
 
/*  ****  Selection of the interval  (binary search within pre-calculated limits). */
      int itn = (int)(ru * (NP_RAYLEIGH-1));     // 'itn' will never reach the last interval 'NP_RAYLEIGH-1', but this is how RITA is implemented in PENELOPE
      int i__ = (int)cgra->itlco[itn + (*mat)*NP_RAYLEIGH];
      int j   = (int)cgra->ituco[itn + (*mat)*NP_RAYLEIGH];
      
      if ((j - i__) > 1)
      {
        do
        {
          register int k = (i__ + j)>>1;     // >>1 == /2 
          if (ru > cgra->pco[k -1 + (*mat)*NP_RAYLEIGH])
            i__ = k;
          else
            j = k;
        }
        while ((j - i__) > 1);
      }
       
/*  ****  Sampling from the rational inverse cumulative distribution. */
      int index = i__ - 1 + (*mat)*NP_RAYLEIGH;

      double rr = ru - cgra->pco[index];
      double xx;
      if (rr > 1e-16)
      {      
        double d__ = (double)(cgra->pco[index+1] - cgra->pco[index]);
        float aco_index = cgra->aco[index], bco_index = cgra->bco[index], xco_index = cgra->xco[index];   // Avoid multiple accesses to the same global variable

        xx = (double)xco_index + (double)(aco_index + 1.0f + bco_index)* d__* rr / (d__*d__ + (aco_index*d__ + bco_index*rr) * rr) * (double)(cgra->xco[index+1] - xco_index);
        
      }
      else
      {
        xx = cgra->xco[index];
      }
      
      if (xx < x2max)
      {
        // Sampled value below maximum possible value:
        *costh_Rayleigh = 1.0 - 2.0 * xx / x2max;   // !!DeBuG!! costh_Rayleigh in double precision, but not all intermediate steps are!?
        /*  ****  Rejection: */    
        if (ranecu_double(seed) < (((*costh_Rayleigh)*(*costh_Rayleigh) + 1.0)*0.5))
          break;   // Sample value not rejected! break loop and return.
      }
    }
} /* graa */



//////////////////////////////////////////////////////////////////////////


//  ***********************************************************************
//  *   Translation of PENELOPE's "SUBROUTINE GCOa"  from FORTRAN77 to C  *
//  ********************************************************************* *
//!  Random sampling of incoherent (Compton) scattering of photons, using 
//!  the sampling algorithm from PENELOPE 2006:
//!    Relativistic impulse approximation with analytical one-electron Compton profiles

//      NOTE: In penelope, Doppler broadening is not used for E greater than 5 MeV.
//            We don't use it in GPU to reduce the lines of code and prevent using COMMON/compos/ZT(M)

//!       @param[in,out] energy   incident and final photon energy (eV)
//!       @param[out] costh_Compton   cosine of the polar scattering angle
//!       @param[in] material   Current voxel material
//!       @param[in] seed   RANECU PRNG seed
//
//  CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
//  C  PENELOPE/PENGEOM (version 2006)                                     C
//  C    Copyright (c) 2001-2006                                           C
//  C    Universitat de Barcelona                                          C
//  C  Permission to use, copy, modify, distribute and sell this software  C
//  C  and its documentation for any purpose is hereby granted without     C
//  C  fee, provided that the above copyright notice appears in all        C
//  C  copies and that both that copyright notice and this permission      C
//  C  notice appear in all supporting documentation. The Universitat de   C
//  C  Barcelona makes no representations about the suitability of this    C
//  C  software for any purpose. It is provided "as is" without express    C
//  C  or implied warranty.                                                C
//  CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
//
//  ************************************************************************
__device__ inline void GCOa(float *energy, double *costh_Compton, int *mat, int2 *seed, struct compton_struct* cgco_SHARED)
{
    float s, a1, s0, af, ek, ek2, ek3, tau, pzomc, taumin;
    float rn[MAX_SHELLS];
    double cdt1;

     // Some variables used in PENELOPE have been eliminated to save register: float aux, taum2, fpzmax, a, a2, ek1 ,rni, xqc, fpz, pac[MAX_SHELLS];

    int i__;
    int my_noscco = cgco_SHARED->noscco[*mat];    // Store the number of oscillators for the input material in a local variable
    

    //!!VERBOSE!!  static int warning_flag_1 = -1, warning_flag_2 = -1, warning_flag_3 = -1;    // Write warnings for the CPU code, but only once.  !!DeBuG!!

    ek = *energy * 1.956951306108245e-6f;    // (1.956951306108245e-6 == 1.0/510998.918)
    ek2 = ek * 2.f + 1.f;
    ek3 = ek * ek;
    // ek1 = ek3 - ek2 - 1.;
    taumin = 1.f / ek2;
    // taum2 = taumin * taumin;
    a1 = logf(ek2);
    // a2 = a1 + ek * 2. * (ek + 1.) * taum2;    // a2 was used only once, code moved below


/*  ****  Incoherent scattering function for theta=PI. */

    s0 = 0.0f;
    for (i__ = 0; i__ < my_noscco; i__++)
    {
       register float temp = cgco_SHARED->uico[*mat + i__*MAX_MATERIALS];
       if (temp < *energy)
       {
         register float aux = *energy * (*energy - temp) * 2.f;
         pzomc = cgco_SHARED->fj0[*mat + i__*MAX_MATERIALS] * (aux - temp * 510998.918f) * rsqrtf(aux + aux + temp * temp) * 1.956951306108245e-6f;
             // 1.956951306108245e-6 = 1.0/510998.918f   // Version using the reciprocal of sqrt in CUDA: faster and more accurate!!
             // ORIGINAL: pzomc = cgco_SHARED->fj0[*mat + i__*MAX_MATERIALS] * (aux - temp * 510998.918f) / (sqrtf(aux + aux + temp * temp) * 510998.918f);
         if (pzomc > 0.0f)
           temp = (0.707106781186545f+pzomc*1.4142135623731f) * (0.707106781186545f+pzomc*1.4142135623731f);
         else
           temp = (0.707106781186545f-pzomc*1.4142135623731f) * (0.707106781186545f-pzomc*1.4142135623731f);

         temp = 0.5f * expf(0.5f - temp);    // Calculate EXP outside the IF to avoid branching

         if (pzomc > 0.0f)
            temp = 1.0f - temp;
                                
         s0 += cgco_SHARED->fco[*mat + i__*MAX_MATERIALS] * temp;
       }
    }
            
/*  ****  Sampling tau. */
    do
    {
      if (ranecu(seed)*/*a2=*/(a1+2.*ek*(ek+1.f)*taumin*taumin) < a1)
      { 
        tau = powf(taumin, ranecu(seed));    // !!DeBuG!!  "powf()" has a big error (7 ULP), the double version has only 2!! 
      }
      else
      {
        tau = sqrtf(1.f + ranecu(seed) * (taumin * taumin - 1.f));
      }

      cdt1 = (double)(1.f-tau) / (((double)tau)*((double)*energy)*1.956951306108245e-6);    // !!DeBuG!! The sampled COS will be double precision, but TAU is not!!!

      if (cdt1 > 2.0) cdt1 = 1.99999999;   // !!DeBuG!! Make sure that precision error in POW, SQRT never gives cdt1>2 ==> costh_Compton<-1
      
  /*  ****  Incoherent scattering function. */
      s = 0.0f;
      for (i__ = 0; i__ < my_noscco; i__++)
      {
        register float temp = cgco_SHARED->uico[*mat + i__*MAX_MATERIALS];
        if (temp < *energy)
        {
          register float aux = (*energy) * (*energy - temp) * ((float)cdt1);

          if ((aux>1.0e-12f)||(temp>1.0e-12f))  // !!DeBuG!! Make sure the SQRT argument is never <0, and that we never get 0/0 -> NaN when aux=temp=0 !!
          {
           pzomc = cgco_SHARED->fj0[*mat + i__*MAX_MATERIALS] * (aux - temp * 510998.918f) * rsqrtf(aux + aux + temp * temp) * 1.956951306108245e-6f;
             // 1.956951306108245e-6 = 1.0/510998.918f   //  Version using the reciprocal of sqrt in CUDA: faster and more accurate!!
             // ORIGINAL: pzomc = cgco_SHARED->fj0[*mat + i__*MAX_MATERIALS] * (aux - temp * 510998.918f) / (sqrtf(aux + aux + temp * temp) * 510998.918f);
          }
          else
          {
            pzomc = 0.002f;    // !!DeBuG!! Using a rough approximation to a sample value of pzomc found using pure double precision: NOT RIGUROUS! But this code is expected to be used very seldom, only in extreme cases.

            //!!VERBOSE!!  if (warning_flag_1<0)
            //!!VERBOSE!!  {  warning_flag_1 = +1;  // Disable warning, do not show again
            //!!VERBOSE!!  // printf("          [... Small numerical precision error detected computing \"pzomc\" in GCOa (this warning will not be repeated).]\n               i__=%d, aux=%.14f, temp=%.14f, pzomc(forced)=%.14f, uico=%.14f, energy=%.7f, cgco_SHARED->fj0=%.14f, mat=%d, cdt1=%.14lf\n", (int)i__, aux, temp, pzomc, cgco_SHARED->uico[*mat+i__*MAX_MATERIALS], *energy, cgco_SHARED->fj0[*mat+i__*MAX_MATERIALS], (int)*mat, cdt1);   // !!DeBuG!!
            //!!VERBOSE!!  }
           
          }
          
          temp = pzomc * 1.4142135623731f;
          if (pzomc > 0.0f)
            temp = 0.5f - (temp + 0.70710678118654502f) * (temp + 0.70710678118654502f);   // Calculate exponential argument
          else
            temp = 0.5f - (0.70710678118654502f - temp) * (0.70710678118654502f - temp);

          temp = 0.5f * expf(temp);      // All threads will calculate the expf together
          
          if (pzomc > 0.0f)
            temp = 1.0f - temp;

          s += cgco_SHARED->fco[*mat + i__*MAX_MATERIALS] * temp;
          rn[i__] = temp;
        }        
      }
    } while( (ranecu(seed)*s0) > (s*(1.0f+tau*(/*ek1=*/(ek3 - ek2 - 1.0f)+tau*(ek2+tau*ek3)))/(ek3*tau*(tau*tau+1.0f))) );  //  ****  Rejection function

    *costh_Compton = 1.0 - cdt1;
        
/*  ****  Target electron shell. */
    for (;;)
    {
      register float temp = s*ranecu(seed);
      float pac = 0.0f;

      int ishell = my_noscco - 1;     // First shell will have number 0
      for (i__ = 0; i__ < (my_noscco-1); i__++)    // !!DeBuG!! Iterate to (my_noscco-1) only: the last oscillator is excited in case all other fail (no point in double checking) ??
      {
        pac += cgco_SHARED->fco[*mat + i__*MAX_MATERIALS] * rn[i__];   // !!DeBuG!! pac[] is calculated on the fly to save registers!
        if (pac > temp)       //  pac[] is calculated on the fly to save registers!  
        {
            ishell = i__;
            break;
        }
      }

    /*  ****  Projected momentum of the target electron. */
      temp = ranecu(seed) * rn[ishell];

      if (temp < 0.5f)
      {
        pzomc = (0.70710678118654502f - sqrtf(0.5f - logf(temp + temp))) / (cgco_SHARED->fj0[*mat + ishell * MAX_MATERIALS] * 1.4142135623731f);
      }
      else
      {
        pzomc = (sqrtf(0.5f - logf(2.0f - 2.0f*temp)) - 0.70710678118654502f) / (cgco_SHARED->fj0[*mat + ishell * MAX_MATERIALS] * 1.4142135623731f);
      }
      if (pzomc < -1.0f)
      {
        continue;      // re-start the loop
      }

  /*  ****  F(EP) rejection. */
      temp = tau * (tau - (*costh_Compton) * 2.f) + 1.f;       // this variable was originally called "xqc"
      
        // af = sqrt( max_value(temp,1.0e-30f) ) * (tau * (tau - *costh_Compton) / max_value(temp,1.0e-30f) + 1.f);  //!!DeBuG!! Make sure the SQRT argument is never <0, and that I don't divide by zero!!

      if (temp>1.0e-20f)   // !!DeBuG!! Make sure the SQRT argument is never <0, and that I don't divide by zero!!
      {
        af = sqrtf(temp) * (tau * (tau - ((float)(*costh_Compton))) / temp + 1.f);
      }
      else
      {
        // When using single precision, it is possible (but very uncommon) to get costh_Compton==1 and tau==1; then temp is 0 and 'af' can not be calculated (0/0 -> nan). Analysing the results obtained using double precision, we found that 'af' would be almost 0 in this situation, with an "average" about ~0.002 (this is just a rough estimation, but using af=0 the value would never be rejected below).

        af = 0.00200f;    // !!DeBuG!!
                
        //!!VERBOSE!!  if (warning_flag_2<0)
        //!!VERBOSE!!  { warning_flag_2 = +1;  // Disable warning, do not show again
        //!!VERBOSE!!    printf("          [... Small numerical precision error detected computing \"af\" in GCOa (this warning will not be repeated)].\n               xqc=%.14f, af(forced)=%.14f, tau=%.14f, costh_Compton=%.14lf\n", temp, af, tau, *costh_Compton);    // !!DeBuG!!
        //!!VERBOSE!!  }

      }

      if (af > 0.0f)
      {
        temp = af * 0.2f + 1.f;    // this variable was originally called "fpzmax"
      }
      else
      {
        temp = 1.f - af * 0.2f;
      }
      
      if ( ranecu(seed)*temp < /*fpz =*/(af * max_value( min_value(pzomc,0.2f) , -0.2f ) + 1.f) )
      {
        break;
      }

    }

/*  ****  Energy of the scattered photon. */
    {
      register float t, b1, b2, temp;
      t = pzomc * pzomc;
      b1 = 1.f - t * tau * tau;
      b2 = 1.f - t * tau * ((float)(*costh_Compton));

      temp = sqrtf( fabsf(b2 * b2 - b1 * (1.0f - t)) );
      
          
      if (pzomc < 0.0f)
         temp *= -1.0f;

      // !Error! energy may increase (slightly) due to inacurate calculation!  !!DeBuG!!
      t = (tau / b1) * (b2 + temp);
      if (t > 1.0f)
      {

        //!!VERBOSE!!  if (warning_flag_3<0)
        //!!VERBOSE!!  { warning_flag_3 = +1;  // Disable warning, do not show again
        //!!VERBOSE!!    printf("\n          [... a Compton event tried to increase the x ray energy due to precision error. Keeping initial energy. (This warning will not be repeated.)]\n               scaling=%.14f, costh_Compton=%.14lf\n", t, *costh_Compton);   // !!DeBuG!!
        //!!VERBOSE!!  }
        
        t = 1.0f; // !!DeBuG!! Avoid increasing energy by hand!!! not nice!!
      }

      (*energy) *= t;
       // (*energy) *= (tau / b1) * (b2 + temp);    //  Original PENELOPE code
    }
    
}  // [End subroutine GCOa]



////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
//!  Tally the depose deposited inside each material.
//!  This function is called whenever a particle suffers a Compton or photoelectric
//!  interaction. The energy released in each interaction is added and later in the 
//!  report function the total deposited energy is divided by the total mass of the 
//!  material in the voxelized object to get the dose. This naturally accounts for
//!  multiple densities for voxels with the same material (not all voxels have same mass).
//!  Electrons are not transported in MC-GPU and therefore we are approximating
//!  that the dose is equal to the KERMA (energy released by the photons alone).
//!  This approximation is acceptable when there is electronic equilibrium and 
//!  when the range of the secondary electrons is shorter than the organ size. 
//!
//!  The function uses atomic functions for a thread-safe access to the GPU memory.
//!  We can check if this tally was disabled in the input file checking if the array
//!  materials_dose was allocated in the GPU (disabled if pointer = NULL).
//!
//!
//!       @param[in] Edep   Energy deposited in the interaction
//!       @param[in] material   Current material id number
//!       @param[out] materials_dose   ulonglong2 array storing the mateials dose [in eV/g] and dose^2 (ie, uncertainty).
////////////////////////////////////////////////////////////////////////////////
__device__ inline void tally_materials_dose(float* Edep, int* material, ulonglong2* materials_dose)
{
  // Note: with many histories and few materials the materials_dose integer variables may overflow!! Using double precision floats would be better. Single precision is not good enough because adding small energies to a large counter would give problems.

  atomicAdd(&materials_dose[*material].x, __float2ull_rn((*Edep)*SCALE_eV) );  // Energy deposited at the material, scaled by the factor SCALE_eV and rounded.
  atomicAdd(&materials_dose[*material].y, __float2ull_rn((*Edep)*(*Edep)) );   // Square of the dose to estimate standard deviation (not using SCALE_eV for std_dev to prevent overflow)
      // OLD:   materials_dose[*material].x += (unsigned long long int)((*Edep)*SCALE_eV + 0.5f);
  return;
}



/* 
   !!inputDensity!! Replacing the hardcoded density_LUT look-up table function with an array in RAM or GPU constant memory:
   OLD LOOK-UP TABLE USED IN VICTRE SIMULATIONS:

////////////////////////////////////////////////////////////////////////////////
//!  Look up table that returns the pre-defined density of the input material.
////////////////////////////////////////////////////////////////////////////////
__device__ __host__    // Function will be callable from host and also from device
inline float density_LUT(int material)                                                  //!!FixedDensity_DBT!! 
{
  float density;
  switch(material)     // Assuming that first material is number 0
  {
    case 0:  // air
      density = 0.0012f;
      break;
    case 1:  // fat
      density  = 0.92f;
      break;
    case 3:  // glandular
      density  = 1.035f;     // - Johns&Yaffe1986: 1.035  ;  Nominal: 1.06;
      break;
    case 10: // Compression Paddle
      density  = 1.06;       //  polystyrene dens = 1.06   ;  PMMA dens = 1.19     !!DBTv1.5!!
      break;
    case 2:  // skin
      density  = 1.090f;
      break;      
    case 4:  // nipple
      density  = 1.090f;  // -> skin?
      break;
//     case 6:  // muscle
//       density  = 1.05f;
//       break;
    case 5:  // ligament(88)
      density  = 1.120f;  // -> connective Woodard?
      break;
//     case  9: // terminal duct lobular unit(95)
//       density  = 1.04f;   // -> muscle?
//       break;
//     case 7:  // duct(125)
//       density  = 1.05f;
//       break;
    case 8:  // artery(150) and vein(225)
      density  = 1.0f;
      break;
    case 11: // Mass/Signal
      density  = 1.06f;      // - Johns&Yaffe1986: Min: 1.027, Mean: 1.044, Max: 1.058  ;  Nominal: 1.06;
      break;
    case 12: // ==Microcalcification
      density  = 1.781f;  // 1.781=0.84*2.12 -> reduced density a factor 0.84 according to: Hadjipanteli et al., Phys Med Biol 62 p 858 (2017)     // Nominal density Calcium_oxalate=2.12
      break;
    case 13: // ==Tungsten edge
      density  = 19.30f;              // !!detectorModel!!
      break;
    case 14: // ==a-Se detector
      density  = 4.50f;              // !!detectorModel!!
      break;
    default:
      density  = 1.05f;   // Using the default value for materials that have the same density.
  }
  
  return density;
}
*/




////////////////////////////////////////////////////////////////////////////////
//!  Tally a radiographic projection image using a detector layer with the input thickness and material composition.
//!  This model will reproduce the geometric spreading of the point spread function and the real detector transmission.
////////////////////////////////////////////////////////////////////////////////
__device__ inline void tally_image(float* energy, float3* position, float3* direction, signed char* scatter_state, unsigned long long int* image, struct source_struct* source_data_SHARED, struct detector_struct* detector_data_SHARED, int2* seed)     //!!detectorModel!!
{
  // Rotate direction to the coordinate system with the detector on XZ plane (Y=0):       // !!DBTv1.4!!
  apply_rotation(direction, detector_data_SHARED->rot_inv);    //!!DBTv1.4!!
  
  
  // Check the angle between the x-ray direction and the Y axis (normal of the detector); return if the particle is moving away from the detector:
  if (direction->y < 0.0175f)
    return;  // Reject particle: angle towards Y axis larger than 89 deg --> particle moving parallel or away from the detector!
  
  // Translate coordinate system to have detector centered at origin:   // !!DBTv1.4!!
  position->x -= detector_data_SHARED->center.x;  position->y -= detector_data_SHARED->center.y;  position->z -= detector_data_SHARED->center.z;

  // Rotate coordinate system to have detector on XZ plane (Y=0):   // !!DBTv1.4!!
  apply_rotation(position, detector_data_SHARED->rot_inv);
  

  
  // Sample the distance to the next interaction in the material of the detector or antiscatter grid protective covers, to determine if the particle will be absorbed in the covers:    !!DBTv1.5!!
  // ASSUMPTIONS: neglecting scattering and fluorescence in the covers; using MFP at average energy spectrum, not the real MFP at current energy.   !!DeBuG!!
  if (detector_data_SHARED->cover_MFP>0.0f)
    if ( (-detector_data_SHARED->cover_MFP*logf(ranecu(seed))) < detector_data_SHARED->cover_thickness )     //  !!DBTv1.5!!
      return;                                                        // Do not tally particle lost in the cover  !!DBTv1.5!!


  // Distance from the particle position to the detector at plane XZ (Y=0):
  float dist_detector = -position->y/direction->y;  

  // Sample and add the extra distance the particle needs to travel to reach the first interaction inside the scintillator (particle not detected if interaction behind thickness):     !!detectorModel!!
  dist_detector += -detector_data_SHARED->scintillator_MFP*logf(ranecu(seed));   // Add distance to next interaction inside the detector material to the detector distance    //!!detectorModel!!
  
  // *** Translate the particle to the detector plane:
  position->x = position->x + dist_detector*direction->x;
  position->y = position->y + dist_detector*direction->y;
  position->z = position->z + dist_detector*direction->z;
  
  if (position->y > detector_data_SHARED->scintillator_thickness)
    return;            // Do not tally energy if particle does not interact inside the detector layer. // !!detectorModel!! !!DBTv1.4!!

  
  // *** Find if particle interacted inside the detector bbox, and compute pixel number (taking into account a possible offset of the detector far from the default centered with the source):
  int pixel_coord_x = __float2int_rd((position->x - detector_data_SHARED->offset.x + 0.5f*detector_data_SHARED->width_X) * detector_data_SHARED->inv_pixel_size_X);  // CUDA intrinsic function converts float to integer rounding down (to minus inf)
  if ((pixel_coord_x>-1)&&(pixel_coord_x<detector_data_SHARED->num_pixels.x))
  {
    int pixel_coord_z = __float2int_rd((position->z - detector_data_SHARED->offset.y + 0.5f*detector_data_SHARED->height_Z) * detector_data_SHARED->inv_pixel_size_Z);
    if ((pixel_coord_z>-1)&&(pixel_coord_z<detector_data_SHARED->num_pixels.y))
    {
      
      // --Sample if the particle is absorbed in the antiscatter grid (scatter or fluorescence in the grid not simulated):
      if (detector_data_SHARED->grid_freq>0.0f)
      {
        if (ranecu(seed) > antiscatter_grid_transmission_prob(position, direction, detector_data_SHARED))         //!!DBTv1.5!!
          return;
      }
   
      // --Sample if all the energy is deposited in the pixel or if a fluorescence x-ray was generated and was able to escape detection:
      //           (k-edge energies available at: http://www.esrf.eu/UsersAndScience/Experiments/StructMaterials/ID11/ID11UserGuide/ID11Edges)
      int flag_fluorescence = 0;
      float edep = *energy;
      if (*energy > detector_data_SHARED->kedge_energy)
      {
        if (ranecu(seed) < detector_data_SHARED->fluorescence_yield)
        { 
          edep -= detector_data_SHARED->fluorescence_energy;       // !!DBTv1.4!! Subtract the input average K fluorescence energy from the deposited energy. The fluorescence photon is simulated afterwards.
          flag_fluorescence = 1;   // !!TrackFluorescence!!
        }
      }

      // -- Particle enters the detector! Tally the particle energy in the corresponding pixel (in tenths of meV):
      //    Using a CUDA atomic function (not available for global floats yet) to read and increase the pixel value in a single instruction, blocking interferences from other threads.
      //    The offset for the primaries or scatter images are calculated considering that:
      //      scatter_state=0 for non-scattered, =1 for Compton, =2 for Rayleigh, and =3 for multiple scatter.
      atomicAdd(( image +                                                               // Pointer to beginning of image array
                (int)(*scatter_state) * detector_data_SHARED->total_num_pixels +         // Offset to corresponding scatter image
                (pixel_coord_x + pixel_coord_z*(detector_data_SHARED->num_pixels.x)) ),  // Offset to the corresponding pixel
                __float2ull_rn(edep*SCALE_eV) );     // Energy arriving at the pixel, scaled by the factor SCALE_eV and rounded.
                                                          // The maximum unsigned long long int value is ~1.8e19:

      // *** Track Fluorescence inside detector:       !!TrackFluorescence!!
      if (flag_fluorescence==1)           
      {
        // -- Sample direction of emission of fluorescence photon isotropically:
        direction->z = 1.0f - 2.0*ranecu(seed);
        float sintheta = sqrtf(1.0f - direction->z*direction->z);
        float phi = (2.0f*PI)*ranecu(seed);
        float cos_phi, sin_phi;
        sincos(phi, &sin_phi, &cos_phi); 
        direction->y = sintheta*sin_phi;
        direction->x = sintheta*cos_phi;
        // -- Sample distance to next fluorescence interaction inside scintillator, using the input MFP at the fluorescence energy:
        dist_detector = -detector_data_SHARED->fluorescence_MFP*logf(ranecu(seed));
        // -- Tally fluorescence energy in the corresponding pixel, unless escaped:        
        position->y = position->y + dist_detector*direction->y;
        if ((position->y>0.0f) && (position->y<detector_data_SHARED->scintillator_thickness))
        {
          position->x = position->x + dist_detector*direction->x;
          pixel_coord_x = __float2int_rd((position->x - detector_data_SHARED->offset.x + 0.5f*detector_data_SHARED->width_X) * detector_data_SHARED->inv_pixel_size_X);  // CUDA intrinsic function converts float to integer rounding down (to minus inf)
          if ((pixel_coord_x>-1)&&(pixel_coord_x<detector_data_SHARED->num_pixels.x))
          {
            position->z = position->z + dist_detector*direction->z;
            pixel_coord_z = __float2int_rd((position->z - detector_data_SHARED->offset.y + 0.5f*detector_data_SHARED->height_Z) * detector_data_SHARED->inv_pixel_size_Z);
            if ((pixel_coord_z>-1)&&(pixel_coord_z<detector_data_SHARED->num_pixels.y))
              atomicAdd(( image + (int)(*scatter_state) * detector_data_SHARED->total_num_pixels + (pixel_coord_x + pixel_coord_z*(detector_data_SHARED->num_pixels.x)) ), __float2ull_rn(detector_data_SHARED->fluorescence_energy*SCALE_eV) );           // !!TrackFluorescence!!
          }
        }
      }

    }
  }
}


////////////////////////////////////////////////////////////////////////////////
//!    Sample two random values with a Gaussian PDF.
//!    Uses the polar method to avoid expensive trigonometric calls implied by the alternative Box-Muller method.
//!         (**Code adapted from penEasyv20140609/penaux.F**)
////////////////////////////////////////////////////////////////////////////////
__device__ inline void gausspdf(float *g1, float *g2, int2 *seed)
{
  float x,y,u;
  do
  {
    x = 1.0f-2.0f*ranecu(seed);
    y = 1.0f-2.0f*ranecu(seed);
    u = x*x+y*y;
  } while ((u>=1.0f)||(u<1.0e-10f));    // Reject point and repeat
  float s = sqrtf(-2.0f*logf(u)/u);
  *g1 = x*s;     // First Gaussian-distributed random variable
  *g2 = y*s;     // Second independent Gaussian-distributed random variable
}


inline void gausspdf_double_CPU(double *g1, double *g2, int2 *seed)
{
  double x,y,u;
  do
  {
    x = 1.0-2.0*ranecu_double_CPU(seed);
    y = 1.0-2.0*ranecu_double_CPU(seed);
    u = x*x+y*y;
  } while ((u>=1.0)||(u<1.0e-10));    // Reject point and repeat
  double s = sqrt(-2.0*log(u)/u);
  *g1 = x*s;     // First Gaussian-distributed random variable
  *g2 = y*s;     // Second independent Gaussian-distributed random variable
}



// ////////////////////////////////////////////////////////////////////////////////
// //!    Return a random value with a Gaussian PDF.                  
// //!    Uses the polar method to avoid expensive trigonometric calls implied by the alternative Box-Muller method.
// //         (**Code adapted from penEasyv20140609/penaux.F**)
// ////////////////////////////////////////////////////////////////////////////////
// __device__ inline float sample_gausspdf(int2 *seed)
// {
//   float x,y,u;
//   do
//   {
//     x = 1.0f-2.0f*ranecu(seed);
//     y = 1.0f-2.0f*ranecu(seed);
//     u = x*x+y*y;
//   } while ((u>=1.0f)||(u<1.0e-10f));    // Reject point and repeat  
//   return (x*sqrtf(-2.0f*logf(u)/u));    // Return Gaussian-distributed random value
// }


////////////////////////////////////////////////////////////////////////////////
//!    Return a random value with a Gaussian PDF, with the distribution cropped at 2 sigma.
//!    Uses the polar method to avoid expensive trigonometric calls implied by the alternative Box-Muller method.
//         (**Code adapted from penEasyv20140609/penaux.F**)
//
//     In a Gaussian distribution, 4.55% of sampled points are farther than 2*sigma; FWHM/2 = sqrt(2*ln(2))*sigma = 1.1774*sigma.
//     Cropping the Gaussian at 2 sigma we prevent generating photons unrealistically far from the focal spot center.
//     Experimental focal spot measurements show that the spot is quite sharp [A Burgess, "Focal spots: I. MTF separability", Invest Radiol 12, p. 36-43 (1977)]
//
////////////////////////////////////////////////////////////////////////////////
__device__ inline float sample_gausspdf_below2sigma(int2 *seed)
{
  float g;
  do                      // Iterate function until we get a value under 2*sigma
  {
    float x,y,u;
    do
    {
      x = 1.0f-2.0f*ranecu(seed);
      y = 1.0f-2.0f*ranecu(seed);
      u = x*x+y*y;
    } 
    while ((u>=1.0f)||(u<1.0e-10f));    // Reject point and repeat

    float s = sqrtf(-2.0f*logf(u)/u);
    g = x*s;              // First Gaussian-distributed random variable    
    
    if (fabsf(g)<2.0f)
      break;              // exit loop and return
    
    g = y*s;              // Second independent Gaussian-distributed random variable
  } 
  while (fabsf(g)>2.0f);
  
  return g;               // Return Gaussian-distributed random value under 2*sigma
}



//!*  Rotate input vector (x,y,z) around the input rotation axis (wx,wy,wz) for the input angle, using Rodrigues' formula to compute the rotation matrix (http://mathworld.wolfram.com/RodriguesRotationFormula.html) :
__device__ __host__ inline void rotate_around_axis_Rodrigues(float *angle, float3 *w, float3 *p)
{
  if (fabs(*angle)>1.0e-8f)  // Apply rotation only if input angle is not 0 
  {
    float s, c;
    sincos(*angle, &s,&c);   // Precompute sinus and cosinus of input angle
    
    float x0 = p->x;         // Temporary copies
    float y0 = p->y;

    // Construct and apply rotation matrix using Rodrigues' formula:
    float m1 =        c+(w->x)*(w->x)*(1-c);   // m1
    float m2 = (w->z)*s+(w->x)*(w->y)*(1-c);   // m4
    float m3 =-(w->y)*s+(w->x)*(w->z)*(1-c);   // m7
    p->x = x0*m1 + y0*m2 +(p->z)*m3;           // x=x0*m1+y0*m4+z0*m7
    
    m1 =-(w->z)*s+(w->x)*(w->y)*(1-c);         // m2
    m2 =        c+(w->y)*(w->y)*(1-c);         // m5
    m3 = (w->x)*s+(w->y)*(w->z)*(1-c);         // m8  
    p->y = x0*m1 + y0*m2 + (p->z)*m3;          // y=x0*m2+y0*m5+z0*m8
    
    m1 = (w->y)*s+(w->x)*(w->z)*(1-c);         // m3
    m2 =-(w->x)*s+(w->y)*(w->z)*(1-c);         // m6
    m3 =        c+(w->z)*(w->z)*(1-c);         // m9
    p->z = x0*m1 + y0*m2 + (p->z)*m3;          // z=x0*m3+y0*m6+z0*m9
  }
}



//!*  Rotate the TWO input vectors (x,y,z) around the input rotation axis (wx,wy,wz) for the input angle, using Rodrigues' formula to compute the rotation matrix (http://mathworld.wolfram.com/RodriguesRotationFormula.html) :
//!*  Rotating the two vectors together I can re-use the rotation matrix computed on the fly
__device__ __host__ inline void rotate_2vectors_around_axis_Rodrigues(float *angle, float3 *w, float3 *p, float3 *v)
{
  if (fabs(*angle)>1.0e-8f)  // Apply rotation only if input angle is not 0 
  {
    float s, c;
    sincos(*angle, &s,&c);   // Precompute sinus and cosinus of input angle
    
    float x0 = p->x, y0 = p->y;         // Temporary copies
    float v0 = v->x, w0 = v->y;

    // Construct and apply rotation matrix using Rodrigues' formula:
    float m1 =        c+(w->x)*(w->x)*(1-c);   // m1
    float m2 = (w->z)*s+(w->x)*(w->y)*(1-c);   // m4
    float m3 =-(w->y)*s+(w->x)*(w->z)*(1-c);   // m7
    p->x = x0*m1 + y0*m2 +(p->z)*m3;           // x=x0*m1+y0*m4+z0*m7
    v->x = v0*m1 + w0*m2 +(v->z)*m3;
    
    m1 =-(w->z)*s+(w->x)*(w->y)*(1-c);         // m2
    m2 =        c+(w->y)*(w->y)*(1-c);         // m5
    m3 = (w->x)*s+(w->y)*(w->z)*(1-c);         // m8  
    p->y = x0*m1 + y0*m2 + (p->z)*m3;          // y=x0*m2+y0*m5+z0*m8
    v->y = v0*m1 + w0*m2 + (v->z)*m3;
    
    m1 = (w->y)*s+(w->x)*(w->z)*(1-c);         // m3
    m2 =-(w->x)*s+(w->y)*(w->z)*(1-c);         // m6
    m3 =        c+(w->z)*(w->z)*(1-c);         // m9
    p->z = x0*m1 + y0*m2 + (p->z)*m3;          // z=x0*m3+y0*m6+z0*m9
    v->z = v0*m1 + w0*m2 + (v->z)*m3;
  }
}


//!*  Rotate the input vector (float3) multiplying by the input rotation matrix (float m[9]).
__device__ __host__ inline void apply_rotation(float3 *v, float *m)
{
  float tmp_x = v->x, tmp_y = v->y;
  v->x = tmp_x*m[0] + tmp_y*m[1] + v->z*m[2];
  v->y = tmp_x*m[3] + tmp_y*m[4] + v->z*m[5];
  v->z = tmp_x*m[6] + tmp_y*m[7] + v->z*m[8];
}





////////////////////////////////////////////////////////////////////////////////
//!  Analytical model of a 1D focused antiscatter grid based on the work of Day and Dance [Phys Med Biol 28, p. 1429-1433 (1983)].
//!  The model returns the probability of transmission through the grid for the current x-ray direction.
//!  The position of the particle in the default reference frame with the detector centered at the origin and laying on the XZ plane is used to compute the focused grid angle.
//!
//!  ASSUMPTIONS: 
//!     - Currently the x-ray energy is not used: the attenuation at the average energy is assumed for every x-ray.          !!DeBuG!!
//!     - Assuming that the focal length of the grid is always identical to the input source-to-detector distance (sdd).     !!DeBuG!!
//!     - The Day and Dance equations are for an uniform oblique grid and the change in angle for consecutive strips is not modeled. As they explain, this is unlikely to be relevant because
//!       the prob of x-rays traversing many strips is extremely low, and consecutive strips have very similar angulation.
//!
//!     - Using double precision for variables that have to be inverted to avoid inaccuracy for collimated rays (u2 close to 0). Using exclusively double takes 4 times more than exclusively floats!
////////////////////////////////////////////////////////////////////////////////
__device__ inline float antiscatter_grid_transmission_prob(float3* position, float3* direction, struct detector_struct* detector_data_SHARED)                   //!!DBTv1.5!!
{
  // -- Compute grid angle at the current location on the detector:
   
  // The default MC-GPU detector orientation is on the XZ plane, perpendicular to Y axis, pointing towards Y. I have to transform to Day&Dance 1983 reference on XY plane, perpendicular to Z axis.
  // The position is already shifted to have the origin at the center of the detector: I can use the position as is to compute the incidence angle -> grid angle for focused grid.
  double grid_angle, u, w;
  if (detector_data_SHARED->grid_ratio<0.0f)
  {
    // <0 --> input orientation == 0 ==> 1D collimated grid with strips perpendicular to lateral direction X (mammo style), as in Day&Dance1983.
    grid_angle = (0.5*PI) - atan2(position->x, detector_data_SHARED->sdd);   // A 0 deg angle between the incident beam and the strips corresponds to a grid angle (sigma) of 90 deg = PI/2
    u = direction->x;
    w = direction->y;
  }
  else
  {   
    // >0 --> input orientation == 1 ==> 1D collimated grid with strips parallel to lateral direction X and perpendicular to Z direction (DBT style): switch Z and X axis
    grid_angle = (0.5*PI) - atan2(position->z, detector_data_SHARED->sdd);
    u = direction->z;
    w = direction->y;
  }

  float C = 1.0f/detector_data_SHARED->grid_freq;
  float d2 = detector_data_SHARED->grid_strip_thickness/sinf(grid_angle);   // Strip thickness in grid reference system   (eq. page 1429, Day&Dance1983)
  float D2 = C - d2;                                   // Distance between consecutive grid strips 
  float h  = fabsf(detector_data_SHARED->grid_ratio) * D2;    // Compute the eight of the grid strips, according to the input grid ratio. Using absolute value bc sign encodes grid orientation in my implementation.
  
  double u2 = fabs(u - w/tan(grid_angle));                // (eq. 1, Day&Dance1983) Note: u2 is the direction RATIO in the oblique referrence system, not the direction COSINE.  
  if (u2<1.0e-9)
    u2 = 1.0e-8;   // !!DeBuG!! Perfectly collimated particles going parallel to strips will have u2=alpha=0. This might gives NaN computing A, but only for few angles (21 deg)??? Add arbitrary epsilon to prevent 0/0.
  
  double P = (h/w)*u2;                                        // (eq. 4, Day&Dance1983)  
  double n = floor(P*detector_data_SHARED->grid_freq);     // grid_freq = 1/C
  float  q = P - n*C;
  double alpha = u2/(detector_data_SHARED->grid_strip_mu-detector_data_SHARED->grid_interspace_mu);   // (eq. 8, Day&Dance1983)
  double inv_alpha = 1.0/alpha;
  

  // Grid transmission: probability of a photon passing through the grid without interaction:
  float A = expf(-detector_data_SHARED->grid_interspace_mu*h/w - d2*n*inv_alpha);    // (eq. 9, Day&Dance1983)  
  float H = 0.0f;   // Step function
  if (q>=D2)
    H = 1.0f;
  float B = (fabsf(q-D2)+2.0f*(float)alpha) * expf((H*(D2-q))*inv_alpha)  +  (fabsf(d2-q)-2.0f*(float)alpha) * expf((-0.5f*(d2+q-fabsf(d2-q)))*inv_alpha);    // (eq. 12, Day&Dance1983)
 
  return (A*B*detector_data_SHARED->grid_freq);     // (eq. 10, Day&Dance1983)     ; grid_freq = 1/C
}





////////////////////////////////////////////////////////////////////////////////
//!  Find the material number at the current location searching the binary tree structure.
//!
//!  @param[in] position  Particle position
//!  @param[in] bitree  Array with the binary trees for every non-uniform coarse voxel
//!  @param[in] bitree_root_index  Index of the root node of the current coarse voxel within bitree array
//!  @param[in] voxel_coord   Voxel coordinates, needed to determine the location of the lower wall of the current coarse voxel
//!  @param[in] voxel_data_CONST.voxel_size  Global variable with the size of the low resolution coarse voxels [cm]
//!  @param[in] voxel_data_CONST.voxel_size_HiRes  Global variable with the size of the original high resolution voxels [cm]
//!  @param[in] voxel_data_CONST.num_voxels_coarse  Global variable with the number of sub-voxels in a coarse voxel
//!  @param[in] voxel_data_CONST.offset  Global variable with the location of the lower walls of the complete voxelized geometry
//!  @return  Material number found at the input position (composition of the tree leaf in that point)
////////////////////////////////////////////////////////////////////////////////
__device__ int find_material_bitree(const float3* position, char* bitree, const int bitree_root_index, short3* voxel_coord)      // !!bitree!! v1.5b
{
  // -- Define variable used during the tree traversal:
  int bitree_node=0, node=0;    // Binary tree node index for the current coarse voxel
  int3 node_width;
  node_width.x = voxel_data_CONST.num_voxels_coarse.x; node_width.y = voxel_data_CONST.num_voxels_coarse.y; node_width.z = voxel_data_CONST.num_voxels_coarse.z;
  float3 node_lower_wall;
  node_lower_wall.x = voxel_coord->x*voxel_data_CONST.voxel_size.x + voxel_data_CONST.offset.x;   // Init lower bound walls in x,y,z
  node_lower_wall.y = voxel_coord->y*voxel_data_CONST.voxel_size.y + voxel_data_CONST.offset.y; 
  node_lower_wall.z = voxel_coord->z*voxel_data_CONST.voxel_size.z + voxel_data_CONST.offset.z;


  // -- Recursively traverse the tree (from the root node) until we get a final node (positive bitree value).
  //    The X, Y, Z axes are divided in half sequentially in each iteration of the loop.
  for(;;)
  {
    bitree_node = (int)bitree[node+bitree_root_index];   // Every acces to the bitree array has to be offset by the root node index "bitree_root_index"
    
    if (bitree_node>-1) // Check if we are already in a final node (empty geometry or final node found in previous Z division):      
      break;   // We reached a final node! Exit infinite loop.
    
    // Negative node value: we need to continue traversing down the tree
    
    // -- Check X axis: 
    if (node_width.x > 1)   // Never split a dimension that is just one voxel wide. Skip splitting and advance to the next dimension.  !!DeBuG!!
    {
      int width_2nd = node_width.x/2;
      node_width.x = node_width.x - width_2nd;  // Integer length of the first node: +1 longer than the second if distance is odd
      float splitting_plane = node_lower_wall.x + node_width.x*voxel_data_CONST.voxel_size_HiRes.x;     // Using the original high res voxel size to determine the location of the splitting plane
      // Check in which side of the middle plane the current position is located:
      if (position->x < splitting_plane)   
      {
        // -Point below (left) the middle plane: move to the following element of the bitree array.
        node++;  // Move to the first child: following node
      }
      else
      {
        // -Point above (right) the middle plane: skip the following subnodes (first half of the node) and move directly to the second half node
        node = -bitree_node;                  // Advance to the location of the 2nd half (stored as a negative value)
        node_lower_wall.x = splitting_plane;  // The splitting plane is now the lower plane of the subnode
        node_width.x = width_2nd;             // Update length of 2nd half subnode
      }  
    }
    
    bitree_node = (int)bitree[node+bitree_root_index];

    if (bitree_node>-1) 
      break;   // We reached a final node! Exit infinite loop.
    
    
    // -- Check Y axis: 
    if (node_width.y > 1)
    {    
      int width_2nd = node_width.y/2;
      node_width.y = node_width.y - width_2nd;
      float splitting_plane = node_lower_wall.y + node_width.y*voxel_data_CONST.voxel_size_HiRes.y;   
      if (position->y < splitting_plane)  
      {
        node++;
      }
      else
      {
        node = -bitree_node;
        node_lower_wall.y = splitting_plane;
        node_width.y = width_2nd;
      }
    }
    
    bitree_node = (int)bitree[node+bitree_root_index];

    if (bitree_node>-1)
      break;   // We reached a final node! Exit infinite loop.
      

    // -- Check Z axis: 
    if (node_width.z > 1)
    {        
      int width_2nd = node_width.z/2;
      node_width.z = node_width.z - width_2nd;
      float splitting_plane = node_lower_wall.z + node_width.z*voxel_data_CONST.voxel_size_HiRes.z;
      if (position->z < splitting_plane)
      {
        node++;
      }
      else
      {
        node = -bitree_node;
        node_lower_wall.z = splitting_plane;
        node_width.z = width_2nd;
      }    
    }
    
  }
  

  // -- We reached a final node: return the material number in the current location
  return (bitree_node);
}
