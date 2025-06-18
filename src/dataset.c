#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "dataset.h"

static Batch *
load_batch ( const char *filepath, const size_t image_size, \
             const size_t num_samples )
{
  FILE *f = fopen( filepath, "rb" );
  if ( !f ) {
    perror( "Failed to open file!" );
    return NULL;
  }

  size_t count = 0, label = 0;
  uint8_t *buffer = malloc( sizeof(uint8_t) * image_size );

  Batch *batch = malloc ( sizeof(Batch) );
  batch->num_samples = num_samples;
  batch->samples = malloc( sizeof(Sample *) * num_samples );
  
  while ( fread(&label, 1, 1, f) == 1 && \
          fread(buffer, 1, image_size, f) == image_size && \
          count < num_samples )
  {
    Sample *sample = malloc( sizeof(Sample) );
    sample->image = malloc( sizeof(double) * image_size );
    for ( size_t i = 0; i < image_size; ++i )
      sample->image[i] = buffer[i] / 255.0;

    sample->label = label;
    batch->samples[count] = sample;
    ++count;
  }
    
  free(buffer);
  fclose(f);

  if ( count != num_samples ) {
    for ( size_t i = 0; i < count; ++i )
      free( batch->samples[i] );
    free( batch );

    fprintf(stderr, "unable to load batch number %zu", count);
    
    return NULL;
  }
  
  return batch;
}

static void
batch_unload ( Batch **batchptr )
{
  if ( batchptr && *batchptr )
  {
    Batch *batch = *batchptr;
    for ( size_t i = 0; i < batch->num_samples; ++i ) {
      free( batch->samples[i] );
      batch->samples[i] = NULL;
    }
    free( batch );
    batchptr = NULL;
  }
}
  
static Batch **
load_batches ( const char *filepath, const char **filenames, \
               const size_t len,     const size_t image_size,
               const size_t num_samples_per_batch )
{
  Batch **batches = malloc( sizeof(Batch *) * len );
  
  for ( size_t i = 0; i < len; ++i ) {
    char filename_buffer[1024];
    snprintf ( filename_buffer, 1024, "%s/%s", filepath, filenames[i] );
    batches[i] = load_batch ( filename_buffer, image_size, \
                              num_samples_per_batch );
  }
  
  return batches;
}

Dataset *
dataset_load_cifar ( const char *filepath )
{
  Dataset * new = malloc( sizeof(Dataset) );
  new->batch_size   = 10000;
  new->image_size   = 32 * 32 * 3;
  new->num_classes  = 10;
  new->failure      = true;
  
  /* load all of the training batches */
  new->train_batches_len = 5;
  const char *train_batches[] = {
    "data_batch_1.bin",
    "data_batch_2.bin",
    "data_batch_3.bin",
    "data_batch_4.bin",
    "data_batch_5.bin",
  };

  new->train_batches = load_batches( filepath, train_batches,
                                     new->train_batches_len,
                                     new->image_size,
                                     new->batch_size );
  
  /* load all (the only one) of the testing batches */
  new->test_batches_len = 1;
  const char *test_batches[] = { "test_batch.bin" };

  new->test_batches = load_batches( filepath, test_batches,
                                    new->test_batches_len,
                                    new->image_size,
                                    new->batch_size );

  char *label_map[] = {
    "airplane", 										
    "automobile", 										
    "bird", 										
    "cat",
    "deer", 										
    "dog", 										
    "frog", 										
    "horse", 										
    "ship", 										
    "truck"
  };

  new->label_map = malloc( sizeof(char *) * new->num_classes );
  for ( size_t i = 0; i < new->num_classes; ++i )
    new->label_map[i] = strdup(label_map[i]);

  new->failure = false;
  
  return new;
}

void
dataset_close ( Dataset **datasetptr )
{
  if ( datasetptr && *datasetptr )
  {
    Dataset *dataset = *datasetptr;

    for ( size_t i = 0; i < dataset->test_batches_len; ++i )
      batch_unload ( &dataset->test_batches[i] );
    free( dataset->test_batches );

    for ( size_t i = 0; i < dataset->train_batches_len; ++i )
      batch_unload ( &dataset->train_batches[i] );
    free( dataset->train_batches );
    
    free( dataset->label_map );

    *datasetptr = NULL;
  }
}
