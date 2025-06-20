#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "model.h"
#include "util.h"

Model *
model_new ( const size_t image_size, const size_t num_classes, float learning_rate )
{
  Model *new = malloc( sizeof(Model) );

  new->weights       = calloc( num_classes * image_size, sizeof(double) );
  new->biases        = calloc( num_classes, sizeof(double) );
  new->image_size    = image_size;
  new->num_classes   = num_classes;
  new->learning_rate = learning_rate;

  model_reset( new );
  
  return new;
}

void
model_reset ( Model *model )
{
  /* initialize the weights with random values */
  for ( size_t i = 0; i < model->num_classes * model->image_size; ++i )
    model->weights[i] = ( rand() / (double) RAND_MAX ) * 0.1 - 0.05;

  memset( model->biases, 0, model->num_classes );
}

void
model_destroy ( Model **model )
{
  if ( model && *model ) {
    free( (*model)->weights );
    free( (*model)->biases );
    free( *model );
    *model = NULL;
  }
}

void
model_train ( Model *model, Dataset *dataset, const size_t epochs )
{ 
  printf( "Beginning training..\n" );
  
  for ( size_t epoch = 0; epoch < epochs; ++epoch ) {
    double total_loss = 0;
    size_t total_samples = 0;
    
    for ( size_t batch_index = 0; batch_index < dataset->train_batches_len; ++batch_index ) {
      Batch *batch = dataset->train_batches[batch_index];

      for ( size_t sample_index = 0; sample_index < batch->num_samples; ++sample_index ) {

        /* generate prediction */
        Sample *sample = batch->samples[sample_index];
        Prediction *pred = model_predict(model, sample);

        if ( sample->image_size != model->image_size )
          printf( "ERR SIZE MISMATCH! %zu != %zu\n", sample->image_size, model->image_size );
        
        ++total_samples;

	static unsigned int error_print_counter = 0;

	if ( error_print_counter < 20 )
	  printf( "Error:          [" );
	
        /* compute losses and gradients */
        for ( size_t class_index = 0; class_index < dataset->num_classes; ++class_index ) {
          bool correct = class_index == sample->label;
          double predicted = clamp( pred->scores[class_index], 1e-9, 1.0 - 1e-9 );

          double error;
          if ( correct ) {
            error = -predicted;
            total_loss += -log( predicted );
          } else
            error = predicted;

	  if (error_print_counter < 20)
	    printf( "% 05.3lf ", error );
          
          /* update weights and biases in the model */
          for ( size_t k = 0; k < sample->image_size; ++k )
            model->weights[ class_index * model->image_size + k ] -= \
              model->learning_rate * error * sample->image[ k ];
          model->biases[ class_index ] -= model->learning_rate * error;
        }

	if ( error_print_counter < 20 )
	  printf("]\n\n");
	
	++error_print_counter;

        prediction_destroy( &pred );
      }
    }

    if (total_samples == 0)
      fprintf(stderr, "no samples were seen!");
    else 
      printf("Epoch %zu/%zu, Samples: %zu, Loss: %.4f\n", epoch + 1, epochs, total_samples, total_loss / total_samples);
  }
}

void
model_test ( Model *model, Dataset *dataset )
{
  printf( "Beginning testing..\n" );
  
  int correct = 0;
  double total_loss = 0.0;
  size_t total_samples = 0;
  float *confusion_matrix = malloc ( sizeof(float) * model->num_classes * model->num_classes );
  
  for ( size_t batch_index = 0; batch_index < dataset->test_batches_len; ++batch_index ) {
    Batch *batch = dataset->test_batches[ batch_index ];
    total_samples += batch->num_samples;
    
    for ( size_t sample_index = 0; sample_index < batch->num_samples; ++sample_index ) {
      /* generate prediction */
      Sample *sample = batch->samples[sample_index];
      Prediction *pred = model_predict( model, sample );
      
      total_loss += -log( pred->scores[ sample->label ] + 1e-9 );
      if ( pred->most_likely == sample->label )
        ++correct;

      confusion_matrix[ sample->label * model->num_classes + pred->most_likely ]++;

      prediction_destroy( &pred );
    }
  }

  float accuracy = 100.0 * correct / total_samples;
  float avg_loss = total_loss / total_samples;

  
  printf("Test Accuracy: %.2f%%\n", accuracy);
  printf("Average Loss: %.4f\n", avg_loss);

  /* TODO: make this graphical so that the actual
           heatmap is displayed with the class names */
  printf("\nConfusion Matrix:\n");
  for ( size_t i = 0; i < model->num_classes; i++ ) {
    for ( size_t j = 0; j < model->num_classes; j++ )
      printf("%.4f ", confusion_matrix[i * model->num_classes + j]);
    printf("\n");
  }
  
}

/* https://en.wikipedia.org/wiki/Softmax_function#Reinforcement_learning */
void
softmax (float *input, float *output, size_t len)
{  
  float max_val = input[0];
  
  for (size_t i = 1; i < len; i++)
    if (input[i] > max_val)
      max_val = input[i];
  
  float sum = 0.0;
  for (size_t i = 0; i < len; i++) {
    output[i] = expf(input[i] - max_val);
    sum += output[i];
  }
    
  for (size_t i = 0; i < len; i++)
    output[i] /= sum;
}

static inline void
print_array ( float *arr, size_t len )
{
  printf("[");
  for ( size_t i = 0; i < len; ++i )
    printf( "% 05.3f ", arr[i] );
  printf("]\n");
}

Prediction *
model_predict ( Model *model, Sample *sample )
{
  static unsigned int debug_counter = 0;
  
  float *scores_raw = calloc ( model->num_classes, sizeof(float) );
  
  Prediction *pred = malloc ( sizeof(Prediction) );
  pred->scores = calloc ( model->num_classes, sizeof(float) );
  pred->num_classes = model->num_classes;
    
  /* generate raw predictions (convert to a probability distribution) */
  for (size_t ci = 0; ci < model->num_classes; ci++) {
    scores_raw[ci] = model->biases[ci];
    for (size_t k = 0; k < sample->image_size; k++)
      scores_raw[ci] += model->weights[ci * model->image_size + k] * sample->image[k];
  }

  /* normalize via softmax */
  softmax(scores_raw, pred->scores, model->num_classes);
  
  /* determine the most likely from the highest score */
  size_t most_likely = 0;
  float highest_score = pred->scores[0];
  for ( size_t i = 1; i < model->num_classes; ++i )
    if ( pred->scores[i] > highest_score ) {
      most_likely = i;
      highest_score = pred->scores[i];
    }
  pred->most_likely = most_likely;

  if ( debug_counter < 20 )
  {
    printf( "True Class:    %zu\n", sample->label );
    printf( "Guessed Class: %zu\n", pred->most_likely );
    printf( "Correct Guess: %s\n", pred->most_likely == sample->label ? "YES" : "NO" );
    
    printf( "Biases (x1000): " );

    float *biases = calloc( model->num_classes, sizeof(float) );
    for (size_t i = 0; i < model->num_classes; ++i)
      biases[i] = (float) model->biases[i] * 1000;
    
    print_array( biases, model->num_classes );
    free(biases);
    
    printf( "Logits:         " );
    print_array( scores_raw, model->num_classes );
    printf( "Probabilities:  " );
    print_array( pred->scores, model->num_classes );
  }
  
  ++debug_counter;
  
  free( scores_raw );
  
  return pred;
}

void
prediction_destroy ( Prediction **pred )
{
  if ( pred && *pred ) {
    free( (*pred)->scores );
    free( *pred );
    *pred = NULL;
  }
}

Model *
model_load_from_file ( const char *filepath )
{
  Model *model = malloc( sizeof(Model) );
  
  FILE* f = fopen(filepath, "rb");
  if (!f) {
    perror("Failed to load model");
    return NULL;
  }

  /* metadata */
  fread(&model->image_size,    sizeof(size_t), 1, f);
  fread(&model->num_classes,   sizeof(size_t), 1, f);
  fread(&model->learning_rate, sizeof(float),  1, f);

  /* data */
  fread(model->weights,        sizeof(double), model->num_classes * model->image_size, f);
  fread(model->biases,         sizeof(double), model->num_classes, f);
  
  fclose(f);

  return model;
}

void
model_save_to_file ( Model *model, const char *filepath )
{
  FILE *f = fopen( filepath, "w" );
  if (!f) {
    perror("Failed to open file for writing");
    return;
  }
  
  /* metadata */
  fwrite(&model->image_size,    sizeof(size_t), 1, f);
  fwrite(&model->num_classes,   sizeof(size_t), 1, f);
  fwrite(&model->learning_rate, sizeof(float),  1, f);

  /* data */
  fwrite(model->weights,        sizeof(double), model->num_classes * model->image_size, f);
  fwrite(model->biases,         sizeof(double), model->num_classes, f);
  
  fclose(f);
}
