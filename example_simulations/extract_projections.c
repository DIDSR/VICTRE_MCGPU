#include <stdio.h>
#include <string.h>
#include <stdlib.h>


//! -- Simple utility to extract the first projection (corresponding to primary+scatter image) from a set 
//!    of MC-GPU projections, such as a DBT dataset, and concatenate the projections in a single raw file.
//!    The input files names must end with the projection number and extension as "_%04d.raw".
//
//     Example execution to combine 25 DBT projections (from 0001 to 0025): 
//      $ time ./extract_projections.x 3000 1500 25 1 mcgpu_image_22183101_scattered
//   
//                                                          [Andreu Badal 2019]


int main(int argc, char **argv)
{
  printf("\n\n\n     *** Extract first projection (primary+scatter) from a set of MC-GPU projections ***\n\n");
  
  if (argc!=6)
  {
    printf("\n     !!ERROR!! Input 5 parameters:  pix_x  pix_y  num_projections  1stProjectionNumber (eg, 1)   InputFileNameRoot (eg, mc-gpu_image.dat)\n\n");
    return -1;
  }

  char input_name[250], file_binary[250];
  
  int i, p, pix_x=atoi(argv[1]), pix_y=atoi(argv[2]), num_projections=atoi(argv[3]), firstProj=atoi(argv[4]);  
  strncpy(input_name, argv[5], 250);
  
  int pixels_per_image = pix_x * pix_y;
  
  float *data = (float*)malloc(pixels_per_image*sizeof(float));   // Allocate space for a complete image
  
  printf("   - Input values:   pix_x=%d, pix_y=%d --> pixels_per_image=%d  ;  num_projections=%d  ;  file name root=%s\n\n", pix_x, pix_y, pixels_per_image, num_projections, input_name);

  strncpy(file_binary, input_name, 250);
  sprintf(file_binary, "%s_%dx%dpixels_%dproj.raw", input_name, pix_x, pix_y, num_projections);
  printf("   - Output binary file: \'%s\'\n\n", file_binary);   //!!VERBOSE!!
  

  FILE* file_binary_out = fopen(file_binary, "wb");
  if (file_binary_out==NULL)
  {
    printf("\n\n   !!fopen ERROR report_image!! Binary file %s can not be opened for writing!!\n", file_binary);
    exit(-3);
  }
  
  // -- Iterate for each projection:
  for(i=firstProj; i<(num_projections+firstProj); i++)
  {    
    sprintf(file_binary, "%s_%04d.raw", input_name, i);   // Create the output file name with the input name + projection number (4 digits, padding with 0)

    printf("   - (%d) Reading file: \'%s\'...\n", i, file_binary);   //!!VERBOSE!!
    fflush(stdout);   // Clear the screen output buffer
    
    FILE* file_binary_in = fopen(file_binary, "rb");  // !!BINARY!!
    if (file_binary_in==NULL)
    {
      printf("\n\n   !!fopen ERROR report_image!! Binary file %s can not be opened for reading!!\n", file_binary);
      exit(-3);
    }

    // -- Read and write data:
    int count = fread(data,  sizeof(float), pixels_per_image, file_binary_in);
    
    if (count!=pixels_per_image)
    {
      printf("\n\n   !!ERROR reading input images!! fread read %d elements, but image has %d pixels!?\n\n", count, pixels_per_image);
      exit(-3);
    }
    else
      fwrite(data, sizeof(float), pixels_per_image, file_binary_out);

    fclose(file_binary_in);
  }

  fclose(file_binary_out); 
  printf("\n\n");
  return 0;
}
