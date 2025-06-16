#ifndef MODEL_HEADER
#define MODEL_HEADER

#include <stdlib.h>
#include "dataset.h"

typedef struct {
  
} Prediction;

typedef struct {
  double **weights, *biases;
} Model;

Model * model_new   ( const size_t image_size, const size_t num_classes );
void    model_train ( Model *model, Dataset *dataset );
void    model_test  ( Model *model, Dataset *dataset );

/* prediction */
void    model_predict ( Model *model, Sample *sample );

/* I/O */
Model * model_load_from_file ( const char *output_filepath );
void    model_save_to_file   ( Model *model, const char *output_filepath );

#endif
