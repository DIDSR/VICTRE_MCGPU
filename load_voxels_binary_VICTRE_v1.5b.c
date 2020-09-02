
////////////////////////////////////////////////////////////////////////////////
//
//              ****************************
//              *** MC-GPU, version 1.5b ***
//              ****************************
//                                          
//!  Auxiliary functions for the VICTRE DBT simulations using a binary geometry
//!  and a binary tree structure: 
//!    a) Read external voxelized phantom as unsigned char values (0 to 255);
//!    b) Assign a Monte Carlo material number to each voxel value;
//!    c) Build the binary tree structure to sort the voxels and reduce memory use in GPU
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
//!                     @file    load_voxels_binary_VICTRE_v1.5b.c
//!                     @author  Andreu Badal (Andreu.Badal-Soler{at}fda.hhs.gov)
//!                     @date    2020/09/01
//
////////////////////////////////////////////////////////////////////////////////



void load_voxels_binary_VICTRE(int myID, char* file_name_voxels, float* density_max, struct voxel_struct* voxel_data, int** voxel_mat_dens_ptr, long long int* voxel_mat_dens_bytes, short int* dose_ROI_x_max, short int* dose_ROI_y_max, short int* dose_ROI_z_max);

// --Binary tree functions:
void create_bitree(int myID, struct voxel_struct* voxel_data, int* voxel_mat_dens, char** bitree, unsigned int* bitree_bytes, int** voxel_geometry_LowRes, unsigned int* voxel_geometry_LowRes_bytes);     //!!bitree!! v1.5b
unsigned long long int subdivide_node(int *node, const int first_node, const int *voxel_coord, const int *node_size, int axis, int subdivision_level, char *bitree, unsigned char *voxels_coarse, int *N_coarse, int *hash_table_counter, unsigned long long int *hash_table_key, int *hash_table_value, const int max_elements_hash_table, const int max_elements_bitree);
unsigned long long int get_hash(int number);
unsigned long long int combine_hash(unsigned long long int hash1, unsigned long long int hash2);
int search_hash_sorted(unsigned long long int hash, const int *hash_table_counter, unsigned long long int *hash_table_key, int *hash_table_value, const int max_elements_hash_table);
void add_hash_sorted(unsigned long long int hash, int node, int insertion_index, int *hash_table_counter, unsigned long long int *hash_table_key, int *hash_table_value, const int max_elements_hash_table);

#define MAX_HASH_SIZE 500000     // Arbitrary limit to the size of the hash tables. This limit might significantly reduce the time spent creating the tree (searching hash tables), at the cost of missing repeated branches to canonicalize



////////////////////////////////////////////////////////////////////////////////
//! Read the original voxel data in binary form and convert to MC material numbers.
//! Input data uses one byte char per voxel.
//
//!       @param[in] file_name_voxels  Name of the voxelized geometry file.
//!       @param[out] density_max  Array with the maximum density for each material in the voxels.
//!       @param[out] voxel_data   Pointer to a structure containing the voxel number and size.
//!       @param[out] voxel_mat_dens_ptr   Pointer to the vector with the voxel materials and densities.
//!       @param[in] dose_ROI_x/y/z_max   Size of the dose ROI: can not be larger than the total number of voxels in the geometry.
////////////////////////////////////////////////////////////////////////////////
void load_voxels_binary_VICTRE(int myID, char* file_name_voxels, float* density_max, struct voxel_struct* voxel_data, int** voxel_mat_dens_ptr, long long int* voxel_mat_dens_bytes, short int* dose_ROI_x_max, short int* dose_ROI_y_max, short int* dose_ROI_z_max)    //!!FixedDensity_DBT!! Allocating "voxel_mat_dens" as a single material integer instead of "float2" (material+density)
{
  MAIN_THREAD if ((strstr(file_name_voxels,".zip")!=NULL)||(strstr(file_name_voxels,".tar.")!=NULL))
    printf("\n\n    -- WARNING load_voxels! The input voxel file name has the extension \'.zip\' or '.tar\'. Only \'.gz\' compression is allowed!!\n\n");     // !!zlib!!
    
  gzFile file_ptr = gzopen(file_name_voxels, "rb");  // Open the file with zlib: the file can be compressed with gzip or uncompressed.   !!zlib!!  
  
  if (file_ptr==NULL)
  {
    printf("\n\n   !! fopen ERROR load_voxels!! File %s does not exist!!\n", file_name_voxels);
    exit(-2);
  }
  MAIN_THREAD 
  {
    printf("\n    -- Reading binary voxel file in RAW format from file \'%s\':\n", file_name_voxels);
    if (strstr(file_name_voxels,".gz")==NULL)
      printf("         (note that MC-GPU can also read voxel and material files compressed with gzip)\n");     // !!zlib!!  
    fflush(stdout);
  }
  
  // -- Store the size of the voxel bounding box (used in the source function):
  voxel_data->size_bbox.x = voxel_data->num_voxels.x * voxel_data->voxel_size.x;
  voxel_data->size_bbox.y = voxel_data->num_voxels.y * voxel_data->voxel_size.y;
  voxel_data->size_bbox.z = voxel_data->num_voxels.z * voxel_data->voxel_size.z;
  
  long long int total_num_voxels = (long long int)voxel_data->num_voxels.x*(long long int)voxel_data->num_voxels.y*(long long int)voxel_data->num_voxels.z;
  
  MAIN_THREAD printf("       Number of voxels in the geometry file (input file parameter):  %d x %d x %d = %lld voxels\n", voxel_data->num_voxels.x, voxel_data->num_voxels.y, voxel_data->num_voxels.z, total_num_voxels);
  MAIN_THREAD printf("       Voxel size (input file parameter):  %f x %f x %f cm  (voxel volume=%e cm^3)\n", voxel_data->voxel_size.x, voxel_data->voxel_size.y, voxel_data->voxel_size.z, voxel_data->voxel_size.x*voxel_data->voxel_size.y*voxel_data->voxel_size.z);
  MAIN_THREAD printf("       Voxel bounding box size:  %f x %f x %f cm\n", voxel_data->size_bbox.x, voxel_data->size_bbox.y,  voxel_data->size_bbox.z);
  
  
  if (*dose_ROI_x_max > -1)   // Check if tally not disabled
  {
    // -- Make sure the input number of voxels in the vox file is compatible with the input dose ROI (ROI assumes first voxel is index 0):
    if ( (*dose_ROI_x_max+1)>(voxel_data->num_voxels.x) || (*dose_ROI_y_max+1)>(voxel_data->num_voxels.y) || (*dose_ROI_z_max+1)>(voxel_data->num_voxels.z) )
    {
      MAIN_THREAD printf("\n       The input region of interest for the dose deposition is larger than the size of the voxelized geometry:\n");
      *dose_ROI_x_max = min_value(voxel_data->num_voxels.x-1, *dose_ROI_x_max);
      *dose_ROI_y_max = min_value(voxel_data->num_voxels.y-1, *dose_ROI_y_max);
      *dose_ROI_z_max = min_value(voxel_data->num_voxels.z-1, *dose_ROI_z_max);
      MAIN_THREAD printf(  "       updating the ROI max limits to fit the geometry -> dose_ROI_max=(%d, %d, %d)\n", *dose_ROI_x_max+1, *dose_ROI_y_max+1, *dose_ROI_z_max+1);         // Allowing the input of an ROI larger than the voxel volume: in this case some of the allocated memory will be wasted but the program will run ok.
    }
    
    if ( (*dose_ROI_x_max+1)==(voxel_data->num_voxels.x) && (*dose_ROI_y_max+1)==(voxel_data->num_voxels.y) && (*dose_ROI_z_max+1)==(voxel_data->num_voxels.z) )
    {
      MAIN_THREAD printf("       The voxel dose tally ROI covers the entire voxelized phantom: the dose to every voxel will be tallied.\n");
    }
    else
      MAIN_THREAD printf("       The voxel dose tally ROI covers only a fraction of the voxelized phantom: the dose to voxels outside the ROI will not be tallied.\n");
  }
  
  // -- Store the inverse of the pixel sides (in cm) to speed up the particle location in voxels.
  voxel_data->inv_voxel_size.x = 1.0f/(voxel_data->voxel_size.x);
  voxel_data->inv_voxel_size.y = 1.0f/(voxel_data->voxel_size.y);
  voxel_data->inv_voxel_size.z = 1.0f/(voxel_data->voxel_size.z);
  
  // -- Allocate the voxel matrix and store array size:
  *voxel_mat_dens_bytes = sizeof(int)*(long long int)voxel_data->num_voxels.x*(long long int)voxel_data->num_voxels.y*(long long int)voxel_data->num_voxels.z;
  *voxel_mat_dens_ptr   = (int*) malloc(*voxel_mat_dens_bytes);                                                           //!!FixedDensity_DBT!!  
    
  unsigned char* voxel_data_buffer = (unsigned char*) malloc(1024*1024);   //!!DBT!!  Buffer to read the raw voxels one MByte at a time
  
  if (*voxel_mat_dens_ptr==NULL || voxel_data_buffer==NULL)
  {
    printf("\n\n   !!malloc ERROR load_voxels!! Not enough memory to allocate %lld voxels (%f Mbytes)!!\n\n", total_num_voxels, (*voxel_mat_dens_bytes)/(1024.f*1024.f));
    exit(-2);
  }

  MAIN_THREAD printf("\n    -- Initializing the voxel material composition array:  (%f Mbytes)...\n", (*voxel_mat_dens_bytes)/(1024.f*1024.f));   //!!FixedDensity_DBT!!
  MAIN_THREAD fflush(stdout);
  
  // -- Read the raw voxel data in small pieces of size (1024*1024) bytes (max 2^31 elements read at once with gzread):
  long long int read_voxels=0, pix0=0;
  unsigned char* voxel_data_buffer_ptr = voxel_data_buffer;
  int v, b, batches = (int)((voxel_data->num_voxels.x*voxel_data->num_voxels.y*(long long int)voxel_data->num_voxels.z)/(long long int)(1024*1024));
  for (b=0; b<batches; b++)
  {
    read_voxels += gzread(file_ptr, voxel_data_buffer_ptr, (1024*1024));    //  !!zlib!!
    // Copy buffer to geometry, casting into integer:
    for (v=0; v<(1024*1024); v++)
    {
      (*voxel_mat_dens_ptr)[pix0]=(int)(*voxel_data_buffer_ptr);
      pix0++;
      voxel_data_buffer_ptr++;
    }
    voxel_data_buffer_ptr = voxel_data_buffer;   // Reset pointer to refill buffer
  }
  int rem = (int)((voxel_data->num_voxels.x*voxel_data->num_voxels.y*(long long int)voxel_data->num_voxels.z)-batches*(long long int)(1024*1024));
  read_voxels += gzread(file_ptr, voxel_data_buffer_ptr, rem);    //  !!zlib!!
  for (v=0; v<rem; v++)
  {
    (*voxel_mat_dens_ptr)[pix0]=(int)(*voxel_data_buffer_ptr);
    pix0++;
    voxel_data_buffer_ptr++;
  }
  
  if (read_voxels!=(voxel_data->num_voxels.x*voxel_data->num_voxels.y*(long long int)voxel_data->num_voxels.z))  // Reading voxels as 1 byte, but storing them in 4 bytes (for bitree pointers)
  {
    printf("\n\n\n   !!malloc ERROR load_voxels!! Error reading the binary voxel values. Wrong num voxels or overflow?? Voxels read = %lld ; expected voxels = %lld\n\n\n", read_voxels, voxel_data->num_voxels.x*voxel_data->num_voxels.y*(long long int)voxel_data->num_voxels.z);
    exit(-2);
  }
  
  // -- Convert input values to Monte Carlo materials, and rotate phantom to have the source rotating around Z:
  //      ** Initial orientation: X=chest-nipple; Y=right-left; Z=feet-head --> Final orientation: X'=feet-head; Y'=Y=right-left; Z'=chest-nipple
  
  int i, j, k;
  for (k=0; k<MAX_MATERIALS; k++)
    density_max[k] = -999.0f;   // Init array with an impossible low density value

  for(k=1; k<=voxel_data->num_voxels.z; k++)
  {
    for(j=1; j<=voxel_data->num_voxels.y; j++)
    {
      for(i=1; i<=voxel_data->num_voxels.x; i++)
      {
        long long int pix = (i-1) + (j-1)*voxel_data->num_voxels.x + (k-1)*(long long int)voxel_data->num_voxels.x*voxel_data->num_voxels.y;      // Input voxel orientation: X=chest-nipple; Y=right-left; Z=feet-head --> Final orientation: X'=feet-head; Y'=Y=right-left; Z'=chest-nipple          

        int input_ID = (int)(*voxel_mat_dens_ptr)[pix];
        
        if (input_ID<0 || input_ID>255)
        {
          printf("\n\n\n   >>>ERROR ASSIGNING VOXEL VALUE>>> Voxel values must be between 0 and 255 (unsigned char). Value read for voxel %lld (%d,%d,%d) = %d \n\n\n",pix,i,j,k,input_ID);
          exit(-2);
        }
        if (voxelId[input_ID]<0)
        {
           printf("\n\n\n   >>>ERROR ASSIGNING VOXEL VALUE>>> Found a voxel number that has not been assigned to any Monte Carlo material in the input file! Value read for voxel %lld (%d,%d,%d) = %d\n\n\n",pix,i,j,k, input_ID);
           exit(-2);
        }

        density_max[voxelId[input_ID]] = -1.0f;     // Flag that this material exists in a voxel of the geometry in binary format.
        
        (*voxel_mat_dens_ptr)[pix] = voxelId[input_ID];   // !!inputDensity!! Assign the required Monte Carlo material composition number to the voxel using the read voxel id and the voxel-to-material conversion table in global memory read from the input file (given after the material file name). The Monte Carlo material numbers are determined by the material file order in the input file.

      }
    }
  }

        // !!VERBOSE!! Output transformed geometry
        //     k=51;
        //     for(j=1; j<=voxel_data->num_voxels.y; j++)
        //     {
        //       for(i=1; i<=voxel_data->num_voxels.x; i++)
        //       {
        //         int pix = (i-1) + (j-1)*voxel_data->num_voxels.x + (k-1)*voxel_data->num_voxels.x*voxel_data->num_voxels.y;
        //         printf("%d\n",(*voxel_mat_dens_ptr)[pix]);
        //       }
        //       printf("\n");
        //     }
        // FILE* file_raw = fopen("geometry_debug.raw", "wb");   //!!VERBOSE!! Output transformed geometry
        // fwrite((*voxel_mat_dens_ptr), sizeof(char), *voxel_mat_dens_bytes, file_raw);   //!!VERBOSE!! Output transformed geometry
        // fclose(file_raw);     //!!VERBOSE!! Output transformed geometry

  
  free(voxel_data_buffer);
  gzclose(file_ptr);     // Close input file    !!zlib!!

}




////////////////////////////////////////////////////////////////////////////////
//! Create a low resolution version of the input voxel data, and create a binary tree for each non-uniform coarse voxel.
//
//!       @param[in] myID  Thread id number (to identify output messages only). myID is used inside the macro "MAIN_THREAD".
//!       @param[in,out] voxel_data   Pointer to a structure containing the voxel number and size.
//!       @param[in,out] voxel_mat_dens   Pointer to the vector with the voxel materials and densities. The high resolution geometry will be changed to a low resolution version
//!       @param[out] bitree   Pointer to the vector storing the binary trees for each coarse voxel
////////////////////////////////////////////////////////////////////////////////
void create_bitree(int myID, struct voxel_struct* voxel_data, int* voxel_mat_dens, char** bitree_ptr, unsigned int* bitree_bytes, int** voxel_geometry_LowRes, unsigned int* voxel_geometry_LowRes_bytes)      //!!bitree!! v1.5b
{
  
  clock_t clock_start = clock();
  
  MAIN_THREAD printf("\n    -- Creating low resolution version of the input voxelized geometry, and binary tree structure for every non-uniform coarse voxels...\n");  
  MAIN_THREAD fflush(stdout);
  
  int3 num_voxels_LowRes;   // Compute the number of low resolution voxels, after dividing by coarse voxels (rounding up)
  num_voxels_LowRes.x = (int)((float)voxel_data->num_voxels.x/(float)voxel_data->num_voxels_coarse.x + 0.99f);                    // !!bitree!! v1.5b
  num_voxels_LowRes.y = (int)((float)voxel_data->num_voxels.y/(float)voxel_data->num_voxels_coarse.y + 0.99f); 
  num_voxels_LowRes.z = (int)((float)voxel_data->num_voxels.z/(float)voxel_data->num_voxels_coarse.z + 0.99f);
  
  unsigned long long int Nvoxels_HiRes = voxel_data->num_voxels.x * (unsigned long long int)(voxel_data->num_voxels.y * voxel_data->num_voxels.z);
  int Nvoxels_coarse = (int)voxel_data->num_voxels_coarse.x * (int)voxel_data->num_voxels_coarse.y * (int)voxel_data->num_voxels_coarse.z;
  unsigned long long int Nvoxels_LowRes = num_voxels_LowRes.x * (unsigned long long int)(num_voxels_LowRes.y * num_voxels_LowRes.z);
  
  // -- Allocate memory for binary tree, low resolution phantom, and hash tables:
  int max_elements_bitree = (int) min_value(2*Nvoxels_coarse*Nvoxels_LowRes, (unsigned long long int)2000000000);  // Limit table to <2^31 elements
  *bitree_bytes = sizeof(char)*max_elements_bitree;  // We are allocating many more nodes than we will end up using; the exact value of bitree_bytes will be updated at the end to allocate the minimum array size in GPU
  *bitree_ptr = (char*) malloc(*bitree_bytes);
  char* bitree = *bitree_ptr;                        // Use a new pointer to avoid using the '*' operator all the time after the allocation
  if (bitree==NULL)
  {
    printf("\n\n   !!malloc ERROR create_bitree!! Not enough memory to allocate %d bitree nodes (%f Mbytes)!!\n\n", max_elements_bitree, (*bitree_bytes)/(1024.f*1024.f));
    exit(-2);
  }
  
  *voxel_geometry_LowRes_bytes = sizeof(int)*Nvoxels_LowRes;
  *voxel_geometry_LowRes = (int*) malloc(*voxel_geometry_LowRes_bytes);     // Allocate array to store the low resolution version of the phantom
  if (*voxel_geometry_LowRes==NULL)
  {
    printf("\n\n   !!malloc ERROR create_bitree!! Not enough memory to allocate the low resolution version of the geometry (%f Mbytes)!!\n\n", (*voxel_geometry_LowRes_bytes)/(1024.f*1024.f));
    exit(-2);
  }

  int max_elements_hash_table = (int) min_value(Nvoxels_coarse*2, MAX_HASH_SIZE);       // Limit the size of the hash table to prevent super-long execution times
  int max_elements_hash_coarse_table = (int) min_value(Nvoxels_LowRes, MAX_HASH_SIZE);
  unsigned long long int *hash_table_key = (unsigned long long int*) malloc(sizeof(unsigned long long int)*(max_elements_hash_table));   // -- Hash table keys for the branches added to the binary tree
  int *hash_table_value = (int*) malloc(sizeof(int)*(max_elements_hash_table));                                                          // -- Hash table positions for the branches added to the binary tree
  unsigned long long int *hash_coarse_table_key = (unsigned long long int*) malloc(sizeof(unsigned long long int)*(max_elements_hash_coarse_table));   // -- Hash table keys for the complete coarse voxels
  int *hash_coarse_table_value = (int*) malloc(sizeof(int)*(max_elements_hash_coarse_table));                                                          // -- Hash table positions for the complete coarse voxels
  unsigned char *voxels_coarse = (unsigned char*) malloc(sizeof(unsigned char)*Nvoxels_coarse);
  if (hash_table_key==NULL || hash_table_value==NULL || hash_coarse_table_key==NULL || hash_coarse_table_value==NULL || voxels_coarse==NULL)
  {
    printf("\n\n   !!!malloc ERROR create_bitree!! Problem allocating memory for hash tables!\n\n");
    exit(-3);
  }

  // -- Initialize data:
  memset(hash_coarse_table_key,  0, sizeof(unsigned long long int)*(max_elements_hash_coarse_table));
  memset(hash_coarse_table_value,0, sizeof(int)*(max_elements_hash_coarse_table));
  memset(bitree,                 0, *bitree_bytes);

  // ** Process each coarse voxel successively and output the coarse voxel matrix and its bitree as output file:
  int z_LowRes, y_LowRes, x_LowRes, i, j, k, node=-1, *node_ptr=&node, max_jump=0, merged_coarse_voxels=0, first_node=0,
      hash_coarse_table_counter=1, *hash_coarse_table_counter_ptr=&hash_coarse_table_counter;
  int N_coarse[3] = {(int)voxel_data->num_voxels_coarse.x, (int)voxel_data->num_voxels_coarse.y, (int)voxel_data->num_voxels_coarse.z};   
  int num_vox_LowRes=0, bitree_counter=0;
  
  for (z_LowRes=0; z_LowRes<num_voxels_LowRes.z; z_LowRes++)
  {
    for (y_LowRes=0; y_LowRes<num_voxels_LowRes.y; y_LowRes++)
    {
      for (x_LowRes=0; x_LowRes<num_voxels_LowRes.x; x_LowRes++)
      {  
        // - Reset global variables for current coarse voxel:
        memset(hash_table_key,  0, sizeof(unsigned long long int)*(max_elements_hash_table));
        memset(hash_table_value,0, sizeof(int)*(max_elements_hash_table));        
        int hash_table_counter=1, *hash_table_counter_ptr=&hash_table_counter;           // The element 0 of the hash table is not used, value stays equal to 0
        memset(voxels_coarse, 0, sizeof(unsigned char)*Nvoxels_coarse);

        // - Copy the data from the original voxel object to the current coarse voxel:
        int flag_nonuniform=0;
        unsigned long long int num_vox_coarse=0;
        for (k=0; k<voxel_data->num_voxels_coarse.z; k++)
        {
          for (j=0; j<voxel_data->num_voxels_coarse.y; j++)
          {
            for (i=0; i<voxel_data->num_voxels_coarse.x; i++)
            { 
              unsigned long long int num_vox_HiRes = ((unsigned long long int)(z_LowRes*voxel_data->num_voxels_coarse.z + k)*voxel_data->num_voxels.x*voxel_data->num_voxels.y) + (unsigned long long int)(y_LowRes*voxel_data->num_voxels_coarse.y + j)*voxel_data->num_voxels.x + (x_LowRes*voxel_data->num_voxels_coarse.x + i);    // Original HiRes voxel number for the current coarse voxel
              
              if ( (z_LowRes*voxel_data->num_voxels_coarse.z + k)>=voxel_data->num_voxels.z || (y_LowRes*voxel_data->num_voxels_coarse.y + j)>=voxel_data->num_voxels.y || (x_LowRes*voxel_data->num_voxels_coarse.x + i)>=voxel_data->num_voxels.x || num_vox_HiRes>=Nvoxels_HiRes )
                voxels_coarse[num_vox_coarse] = (unsigned char)10;   // Avoid overflow. This part of the coarse voxel is outside the original HiRes voxel geometry. This sub-voxel will never be used; assigned value irrelevant
              else
              {
                voxels_coarse[num_vox_coarse] = voxel_mat_dens[num_vox_HiRes];                                                        // Assign the composition of the original geometry to the coarse voxel
                if(0==flag_nonuniform && voxels_coarse[num_vox_coarse]!=voxels_coarse[0])
                  flag_nonuniform = 1;     // Detect that there is more than one kind of voxel inside this coarse voxel
              }              
              num_vox_coarse++;
            }
          }
        }        
        if(0==flag_nonuniform)
        {
          (*voxel_geometry_LowRes)[num_vox_LowRes] = (int)voxels_coarse[0];   // ==> Report material number for current coarse voxel as a positive (or 0) value 
        }
        else
        {
          // -- Create binary tree recursively from the voxelized object root node (voxels bounding box):
          bitree_counter++;
          int voxel_coord[3]={0,0,0}, node_size[3]={voxel_data->num_voxels_coarse.x, voxel_data->num_voxels_coarse.y, voxel_data->num_voxels_coarse.z};
          int max_tmp=-1, axis=-1;


          // **** BUILD BINARY TREE RECURSIVELY. The variable "node" keeps increasing for each successive coarse voxel tree ****
          unsigned long long int hash_coarse = subdivide_node(node_ptr, first_node, voxel_coord, node_size, axis, 0, bitree, voxels_coarse, N_coarse, hash_table_counter_ptr, hash_table_key, hash_table_value, max_elements_hash_table, max_elements_bitree);


        int previous_first_node = search_hash_sorted(hash_coarse, hash_coarse_table_counter_ptr, hash_coarse_table_key, hash_coarse_table_value, max_elements_hash_coarse_table);
        if (previous_first_node<0) 
        {
          // -- Store and report brand new coarse voxel:
          add_hash_sorted(hash_coarse, first_node, -previous_first_node, hash_coarse_table_counter_ptr, hash_coarse_table_key, hash_coarse_table_value, max_elements_hash_coarse_table); 
          int b;
          for(b=first_node; b<=node; b++) 
          {
//             fwrite(&bitree[b], sizeof(int), 1, file_ptr_bitree);   // ==> Report binary tree for current coarse voxel. Negative values indicate location of right side nodes, relative to sub-tree root node (less bits needed)            
            if (bitree[b]<0 && abs(bitree[b])>max_tmp)
              max_tmp = abs(bitree[b]);     // Detect longest internal jump among all bitrees (will determine the bits required for bitree)
          }
        }
        else
        {
          // -- Repeated coarse voxel! Delete repeated info and point to previous bitree location: 
          merged_coarse_voxels++;
          node = first_node-1;
          first_node = previous_first_node;
        }

        (*voxel_geometry_LowRes)[num_vox_LowRes] = -first_node;        // ==> Report pointer to starting location of bitree for current coarse voxel, as a negative value

          if (max_tmp>max_jump)
            max_jump = max_tmp;
          if (max_jump>128)
          {
            printf("\n\n   !!!create_bitree!! ERROR: Coarse voxel too large, too many subdivisions! Not possible to store bitree in a signed char type: max_jump=%d.\n", max_jump);
            printf(    "                             Reduce the number of subdivisions in the input file until max_jump is not larger than 128. \n\n");
            exit(-1);
          }

          first_node = node+1;
        }

        num_vox_LowRes++;
      }
    }
  }

  // Update the number of bytes required to store the bitree, and shrink pre-allcoated array to save memory
  *bitree_bytes = sizeof(char)*(node+1);
  *bitree_ptr = (char*) realloc(*bitree_ptr, *bitree_bytes);
  
  MAIN_THREAD printf("  !!bitree!!  Number of low resolution voxels: %llu\n", Nvoxels_LowRes);
  MAIN_THREAD printf("              Number of non-uniform low resolution voxels converted to binary trees: %d  (%f%%)\n", bitree_counter, 100.0f*bitree_counter/(float)Nvoxels_LowRes);
  MAIN_THREAD printf("              Total number of binary tree nodes stored in memory for all trees: %d\n", node+1);
  MAIN_THREAD printf("              Identical non-uniform low resolution voxels merged: %d  (%f%%)\n", merged_coarse_voxels, 100.0f*merged_coarse_voxels/(float)bitree_counter);   // (MAX_HASH_SIZE==%d), MAX_HASH_SIZE);
  MAIN_THREAD printf("              Longest internal jump among all binary trees (max jump=128): %d\n\n", max_jump);
  
  
  //  -- Changing important geometric parameters below: the low resolution voxelized geometry will be used by default, only the bitree gives info on the higher res details.    !!DeBuG!! 
  
  // -- Update the variable "num_voxels" to the size of the low resolution phantom: only the bitree will have the complete high res information!      !!bitree!! 
  MAIN_THREAD printf("              The geometric parameters \'num_voxels\' and \'voxel_size\' have been updated to the size of the low res voxel geometry: only the bitree has the original high res information.\n");
  MAIN_THREAD printf("                 voxel_data->num_voxels.x=%d (before: %d) , voxel_data->num_voxels.y=%d (before: %d) , voxel_data->num_voxels.z=%d (before: %d)\n", num_voxels_LowRes.x, voxel_data->num_voxels.x, num_voxels_LowRes.y, voxel_data->num_voxels.y, num_voxels_LowRes.z, voxel_data->num_voxels.z);
  MAIN_THREAD printf("                 voxel_data->voxel_size.x=%f cm (before: %f) , voxel_data->voxel_size.y=%f cm (before: %f) , voxel_data->voxel_size.z=%f cm (before: %f)\n\n", voxel_data->voxel_size.x*voxel_data->num_voxels_coarse.x, voxel_data->voxel_size.x, voxel_data->voxel_size.y*voxel_data->num_voxels_coarse.y, voxel_data->voxel_size.y, voxel_data->voxel_size.z*voxel_data->num_voxels_coarse.z, voxel_data->voxel_size.z);
  
  voxel_data->num_voxels.x = num_voxels_LowRes.x;
  voxel_data->num_voxels.y = num_voxels_LowRes.y;
  voxel_data->num_voxels.z = num_voxels_LowRes.z;
  
  voxel_data->voxel_size_HiRes.x = voxel_data->voxel_size.x;     // Save the original high resolution voxel size, necessary to determine the splitting planes of the the binary tree (function "find_material_bitree").
  voxel_data->voxel_size_HiRes.y = voxel_data->voxel_size.y;
  voxel_data->voxel_size_HiRes.z = voxel_data->voxel_size.z;
  
  voxel_data->voxel_size.x = voxel_data->voxel_size.x * (float)voxel_data->num_voxels_coarse.x;
  voxel_data->voxel_size.y = voxel_data->voxel_size.y * (float)voxel_data->num_voxels_coarse.y;
  voxel_data->voxel_size.z = voxel_data->voxel_size.z * (float)voxel_data->num_voxels_coarse.z;  
  voxel_data->inv_voxel_size.x = 1.0/voxel_data->voxel_size.x;
  voxel_data->inv_voxel_size.y = 1.0/voxel_data->voxel_size.y;
  voxel_data->inv_voxel_size.z = 1.0/voxel_data->voxel_size.z;  
  //NOTE: "voxel_data->size_bbox" is not updated: a binary tree might go outside the original bbox but we don't need to track particles in there     //!!DeBuG!! 
  
  printf("       Time spent creating the binary trees and the low res voxel geometry = %.3f s.\n\n", (double)(clock()-clock_start)/CLOCKS_PER_SEC);
  

// FILE* file_raw = fopen("bitree_debug.raw", "wb");                       //!!VERBOSE!! Output bitree
// fwrite(*voxel_geometry_LowRes, sizeof(int), Nvoxels_LowRes, file_raw);  //!!VERBOSE!! Output bitree
// fclose(file_raw);                                                       //!!VERBOSE!! Output bitree  
  
}






////////////////////////////////////////////////////////////////////////////////
//!  Recursively subdivide voxelized object into a binary tree.
//  The "bitree" array stores the tree with the following format: 0 or positive value indicates the material of the final leave; 
//  a value <0 indicates the location in the array where we can find the right side of the node. 
//  The location is provided relative to the index of the "first_node" in the current coarse voxel tree. This means that the value 
//  "first_node" has to be added to the absolute value of "bitree" to find the element in the array that has the information on the righ subnode.
//  The left subnodes are located always in the following array element.
//
////////////////////////////////////////////////////////////////////////////////
unsigned long long int subdivide_node(int *node, const int first_node, const int *voxel_coord, const int *node_size, int axis, int subdivision_level, char *bitree, unsigned char *voxels_coarse, int *N_coarse, int *hash_table_counter, unsigned long long int *hash_table_key, int *hash_table_value, const int max_elements_hash_table, const int max_elements_bitree)
{
  unsigned long long int my_hash = (unsigned long long int)0;
    
  (*node)++;
  int my_node = *node;
  subdivision_level++;

  if ((*node)>max_elements_bitree)
  {
    printf("[subdivide_node] !!ERROR!! Number of nodes exceeds allocated capacity of bitree array!! Do we need long long int to represent the number of tree nodes?");   // !!VERBOSE!!
    return 0;
  }
  

  // -- Check if final node, i.e., single voxel:
  if (1==node_size[0] && 1==node_size[1] && 1==node_size[2])
  {
    unsigned long long int absvox = voxel_coord[2]*(unsigned long long int)(N_coarse[1]*N_coarse[0]) + (unsigned long long int)(voxel_coord[1]*N_coarse[0] + voxel_coord[0]);  // Get voxel composition and size from global variables; handle the case where absvox>2^31 using 64 bits
    int material = (int)voxels_coarse[absvox];   //NOTE: material=0 is possible
    my_hash = get_hash(material);

    if (material>127)
    {
      printf("[subdivide_node] !!ERROR!! A material number >127 was found: not possible to create the binary tree structure with char data type!!!!  material=%d, my_node=%d\n\n\n", material, my_node);   // !!DeBuG!!
      exit(-1);
    }

    bitree[my_node] = material;     // Return the material found in this final node

  }
  else
  {
    // -- Process non-final node:
    
    //    * Set next axis to divide (options: x=0,y=1,z=2)
    axis = (2!=axis)? (axis+1) : 0;
    if (1==node_size[axis]) axis = (2!=axis)? (axis+1) : 0;   // Skip dividing a single-voxel axis
    if (1==node_size[axis]) axis = (2!=axis)? (axis+1) : 0;   // Skip a second single-voxel axis, if necessary


    //    * Subdivide first half node:
    int new_voxel_coord[3]={voxel_coord[0],voxel_coord[1],voxel_coord[2]};   // Create local copies before recursive call
    int new_node_size[3]={node_size[0],node_size[1],node_size[2]};
    int width_2nd = (int)new_node_size[axis]/2;         // Length second subnode
    new_node_size[axis] = new_node_size[axis] - width_2nd;  // Length first node after subdivision (1 longer than the second if odd size)
    unsigned long long int hash_1st = subdivide_node(node, first_node, new_voxel_coord, new_node_size, axis, subdivision_level, bitree, voxels_coarse, N_coarse, hash_table_counter, hash_table_key, hash_table_value, max_elements_hash_table, max_elements_bitree);


    //    * Subdivide second half node (if size>0):
    unsigned long long int hash_2nd = 0;
    new_voxel_coord[axis] = new_voxel_coord[axis]+new_node_size[axis];
    new_node_size[axis] = width_2nd;  // Update new node size for 2nd subnode
    int node_2nd_start = (*node)+1;
    if (width_2nd>0)
    {
      hash_2nd = subdivide_node(node, first_node, new_voxel_coord, new_node_size, axis, subdivision_level, bitree, voxels_coarse, N_coarse, hash_table_counter, hash_table_key, hash_table_value, max_elements_hash_table, max_elements_bitree);
    }
    else
    {
      // This sub-node of size 0 will never be accessed! I init it in a way that it might be pruned below if 1st side is a leave.
      bitree[node_2nd_start] = bitree[*node];
      (*node)++;
    }



    //    * Check if both subnodes have same material, and fuse them into one:
    //      (Disabling the following if statement will disable the essential node pruning, and create a complete tree with as many leaves as voxels.)
    if (2==(*node-my_node) && bitree[my_node+1]==bitree[my_node+2])  // The current node contains two final sub-nodes (leaves) with the same material: join them into one, and delete sub-nodes
    {
      bitree[my_node] = bitree[my_node+1];
      bitree[my_node+1] = bitree[my_node+2] = 0;
      *node = my_node;
      my_hash = hash_1st;                           // Both sides should have same hash in this case
    }
    else
    {

      if ((node_2nd_start - first_node)>128)
      {
        printf("[subdivide_node] !!ERROR!! A jump >128 was found in a binary tree: not possible to create the binary tree structure with char data type!!!!  (node_2nd_start-first_node)=%d, my_node=%d\n\n\n", (node_2nd_start-first_node), my_node);   // !!DeBuG!!
        exit(-1);
      }
      
      bitree[my_node] = -1*(node_2nd_start - first_node);      // Assign link to second subnode (without canonicalization at this moment), relative to the starting node value. Negative sign used to indicate a link
      
      
// COMMENT CODE BELOW OR DEFINE PREPROCESSOR PARAMETER "DISABLE_CANON" TO DISABLE BRANCH CANONICALIZATION:
#ifndef DISABLE_CANON      
        // Use the hash table to detect repeated tree branches at the tree level larger than the input minimum level:
        if (width_2nd>0)
        {
          int previous_2nd_branch = search_hash_sorted(hash_2nd, hash_table_counter, hash_table_key, hash_table_value, max_elements_hash_table);

          if (previous_2nd_branch>=0)   // Duplicated hash found in list: apply canonicalization
          {

            bitree[my_node] = -1*(previous_2nd_branch - first_node);
            *node = node_2nd_start - 1;                   // Move the counter to the end of the first half, because the second half is unnecessary
          }
          else
          {
            add_hash_sorted(hash_2nd, node_2nd_start, -previous_2nd_branch, hash_table_counter, hash_table_key, hash_table_value, max_elements_hash_table);
          }
        }

        int previous_1st_branch = search_hash_sorted(hash_1st, hash_table_counter, hash_table_key, hash_table_value, max_elements_hash_table);
        if (previous_1st_branch<0) 
          add_hash_sorted(hash_1st, my_node+1, -previous_1st_branch, hash_table_counter, hash_table_key, hash_table_value, max_elements_hash_table);  // Store new left branch in case there is an identical right branch downstream.
#endif
      
      my_hash = combine_hash(hash_1st, hash_2nd);    // Combine two hashes into one.
    }
  }
  return my_hash;  
}



//! Generate an initial hash value based on the input material number. 
// The returned value is not random at all, but the heuristic equation tries to separate the values for close numbers 
// and use many different bits of the 64-bit hash, while making sure different materials never return the same hash or 0:
unsigned long long int get_hash(int number)
{
  unsigned long long int hash = (unsigned long long int)((number+5)*(number+17)) + (((unsigned long long int)((number+2)*(number+13)))<<32);
  return hash;
}




//! Combine the two input hashes using the Tiny Encryption Algorithm (TEA) with hash1 as message and hash2 as key.
//  The TEA code was released into the public domain by David Wheeler and Roger Needham (https://dx.doi.org/10.1007%2F3-540-60590-8_29).
//  For more details see: https://en.wikipedia.org/wiki/Tiny_Encryption_Algorithm
unsigned long long int combine_hash(unsigned long long int hash1, unsigned long long int hash2)
{  
  unsigned int v0=(unsigned int)hash1, v1=(unsigned int)(hash1>>32), sum=0, i;    /* set up */
  unsigned int delta=0x9e3779b9;                                                  /* a key schedule constant */
  unsigned int k0=(unsigned int)(hash2),     k1=(unsigned int)(hash2>>8),         /* cache key */
               k2=(unsigned int)(hash2>>16), k3=(unsigned int)(hash2>>32);  // The 4 integers of the key are created by shifting the different bytes of the second hash
  for (i=0; i < 32; i++) {                                                        /* basic cycle start */
      sum += delta;
      v0 += ((v1<<4) + k0) ^ (v1 + sum) ^ ((v1>>5) + k1);
      v1 += ((v0<<4) + k2) ^ (v0 + sum) ^ ((v0>>5) + k3);
  }                                                                               /* end cycle */
  unsigned long long int combined_hash = ((unsigned long long int)v0) + (((unsigned long long int)v1)<<32);  
  return combined_hash;
}
  


//! Search the hash table to determine if an identical tree branch was already stored in memory.
//! If a match is NOT found, the return value is negative (and !=0), and its absolute value gives the 
//! location in the hash list where the current hash must be inserted to keep a sorted array.
//! If the hash is already in the list, the returned value gives the previous node index (>=0).
int search_hash_sorted(unsigned long long int hash, const int *hash_table_counter, unsigned long long int *hash_table_key, int *hash_table_value, const int max_elements_hash_table)
{ 
  int node_number;
  
  // - Binary search in a sorted list:
  int first  = 0,
      last   = (*hash_table_counter), 
      middle = (first+last)/2;

  while (first <= last) 
  {
    if (hash_table_key[middle] < hash)
      first = middle + 1;
    else if (hash_table_key[middle] == hash)
      break;            // Hash found in table: return the location where the hash is found
    else
      last = middle - 1;

    middle = (first + last)/2;
  }

  if (first > last)
    node_number = -(middle+1);                 // Return the location in the hash list where the current hash has to be inserted to keep a sorted array, as a negative value
  else
    node_number = hash_table_value[middle];    // Return the index of older bitree node with the same hash

  return node_number;
}


//! Add a hash value uniquely indentifying a tree branch, and the corresponding node index, to the hash list.
//! The insertion location of the current hash has to be determined with a previous call to "search_hash_sorted"
//! in order to keep a sorted list that can be quickly searched by a binary search. New elements are added in 
//! the list at the corresponding location according to the hash value, and all the elements with a larger hash
//! are shifted one location up. When the maximum number of elements is reached, old values are overwritten
//! (we hope that recent nodes are more likely to be a match than older ones).
void add_hash_sorted(unsigned long long int hash, int node, int insertion_index, int *hash_table_counter, unsigned long long int *hash_table_key, int *hash_table_value, const int max_elements_hash_table)
{
  // ** Check if new elements can be added to the hash table, or if an old entry has to be overwritten:
  if ((*hash_table_counter)<(max_elements_hash_table-1))
  { 
    // -- Insert new value keeping list sorted:
    (*hash_table_counter)++;
    
    if ((*hash_table_counter)>2)  // Do not shift list if we encountered the first element: position = 1, 0 is not used
    {
      int i;
      for (i=(*hash_table_counter); i>insertion_index; i--)
      {
        hash_table_key[i]   = hash_table_key[i-1];      // Shift all elements above the insertion place by one position
        hash_table_value[i] = hash_table_value[i-1];
      }
    }
    hash_table_key[insertion_index]   = hash;    // Insert new value in the proper position in the sorted list
    hash_table_value[insertion_index] = node;
  }
  else
  {
    // -- Replace old value in the current location:
    hash_table_key[insertion_index]   = hash;
    hash_table_value[insertion_index] = node;
  }
  return;
}







// // --Test main program:
// int main(int argc, char **argv)
// {  
//   char file_name_voxels[90];
//   strcpy(file_name_voxels, "/raidb/VICTRE/prePilot/phantoms/lesionPhantoms/pcl_1022999306_1051_1657_713.raw");
//   if (argc==2)
//   {
//     strcpy(file_name_voxels, argv[1]);
//   }
//   
//   int myID=0;
//   float density_max[MAX_MATERIALS];
//   struct voxel_struct voxel_data;
//   char *voxel_mat_dens_ptr;
//   long long int voxel_mat_dens_bytes;
//   short int dose_ROI_x_max=10, dose_ROI_y_max=10, dose_ROI_z_max=10;
//     
//   load_voxels_binary_VICTRE(myID, file_name_voxels, density_max, &voxel_data, &voxel_mat_dens_ptr, &voxel_mat_dens_bytes, &dose_ROI_x_max, &dose_ROI_y_max, &dose_ROI_z_max);
// 
//   return 0;
// }



