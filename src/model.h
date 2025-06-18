#ifndef MODEL_HEADER
#define MODEL_HEADER

#include <stdlib.h>
#include <stdbool.h>
#include "dataset.h"

typedef struct {
  float *scores;
  
  /* vector sizing */
  size_t num_classes, most_likely;

  /* for error handling */
  bool failure;
} Prediction;

typedef struct {
  double *weights, *biases;
  size_t image_size, num_classes;
  float learning_rate;
} Model;

Model *model_new     ( const size_t image_size, const size_t num_classes, \
                      float learning_rate );
void   model_destroy ( Model **model );
void   model_train   ( Model *model, Dataset *dataset, const size_t epochs );
void   model_test    ( Model *model, Dataset *dataset );

/* prediction */
Prediction * model_predict      ( Model *model, Sample *sample );
void         prediction_destroy ( Prediction **pred );

/* I/O */
Model * model_load_from_file ( const char *filepath );
void    model_save_to_file   ( Model *model, const char *filepath );

#endif
