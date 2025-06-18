#ifndef DATASET_HEADER
#define DATASET_HEADER

#include <stdbool.h>

typedef struct {
  double *image;
  size_t label, image_size;
} Sample;

typedef struct {
  Sample **samples;
  size_t num_samples;
} Batch;

typedef struct {
  /* data */
  Batch **test_batches,    **train_batches;
  size_t  test_batches_len,  train_batches_len;

  /* dataset metadata */
  size_t batch_size, image_size, num_classes;
  char **label_map;

  /* did the load fail */
  bool failure;
} Dataset;

Dataset *dataset_load_cifar ( const char *root_filepath );
void     dataset_close      ( Dataset **data );

#endif
