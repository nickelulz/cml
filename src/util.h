#ifndef UTILITY_HEADER
#define UTILITY_HEADER

inline double clamp(double d, double min, double max) {
  const double t = d < min ? min : d;
  return t > max ? max : t;
}

inline void
print_array ( float *arr, size_t len )
{
  printf("[");
  for ( size_t i = 0; i < len; ++i )
    printf( "% 07.3f ", arr[i] );
  printf("]\n");
}

#endif
