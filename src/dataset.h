#ifndef DATASET_HEADER
#define DATASET_HEADER

typedef struct {
  double *image;
  const size_t label, image_size;
} Sample;

typedef struct {
  Sample **samples;
  size_t num_samples;
} Batch;

typedef struct {
  /* data */
  Batch *test_batches,    *train_batches;
  size_t test_batches_len, train_batches_len;

  /* dataset metadata */
  const size_t batch_size, image_size, num_classes, num_features;
  char **label_map;
} Dataset;

Dataset *dataset_load_cifar ( const char *root_filepath );
void     dataset_close      ( Dataset **data );

#endif
