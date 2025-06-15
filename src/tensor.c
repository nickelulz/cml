#include <string.h>
#include "tensor.h"

/* basic operations */
Tensor2D *
Tensor2D_create ( size_t rows, size_t cols )
{
  Tensor2D *t = (Tensor2D *) malloc(sizeof(Tensor2D));
  t->data = malloc(sizeof(double) * rows * cols);
  t->rows = rows;
  t->cols = cols;
  return t;
}

Tensor2D *
Tensor2D_copy ( Tensor2D * t )
{
  Tensor2D *new = Tensor2D_create(t->rows, t->cols);
  new->data = memcpy(new->data, t->data, sizeof(t->data));
  return new;
}
  
void
Tensor2D_destroy ( Tensor2D *t )
{
  free(t->data);
  free(t);
  &t = NULL;
}

/* actual math operations */
Tensor2D *
Tensor2D_transpose ( Tensor2D *t )
{
  Tensor2D *new = Tensor2D_create(t->cols, t->rows);
  for (size_t r = 0; r < t->rows; ++r)
    for (size_t c = 0; c < t->cols; ++c)
      Tensor2D_set_index(new, c, r, Tensor2D_get_index(t, r, c));
  return new;
}

Tensor2D *
Tensor2D_mult ( Tensor2D *a, Tensor2D *b )
{
  if (a->cols != b->rows)
    return NULL;
  
  Tensor2D *result = Tensor2D_create ( b->cols, a->rows );
  
  for (size_t i = 0; i < a->rows; ++i)
    for (size_t j = 0; j < b->cols; ++j)
      for (size_t k = 0; k < a->cols; ++k)
        Tensor2D_set_index( result, i, j,
                            Tensor2D_get_index( a, i, k ) *
                            Tensor2D_get_index( b, k, j ) );

  return result;
}

Tensor2D *
Tensor2D_sq_inverse ( Tensor2D *t )
{
  /* tensor must be square */
  if (t->rows != t->cols)
    return NULL;

  /* gauss-jordan elimination to calculate the inverse
     https://en.wikipedia.org/wiki/Gaussian_elimination */
  Tensor2D *intermediate_augment = Tensor2D_create( t->rows, t->cols + t->rows );

  /* perform calculations */
  
  
  /* copy data */
  Tensor2D *sq_inv_result = Tensor2D_create ( t->rows, t->cols );
  for (size_t r = 0; r < t->rows; ++r)
    for (size_t c = 0; c < t->cols; ++c)
      Tensor2D_set_index ( sq_inv_result, r, c,
                           Tensor2D_get_index ( intermediate_augment, r + t->rows, c ) );
  
  /* free intermediate and return */
  Tensor2D_destroy( intermediate_augment );
  return sq_inv_result;
}

 /* data manipulation */
void
Tensor2D_fill_column ( Tensor2D *t, const size_t col, const double val )
{
  for (size_t r = 0; r < t->rows; ++r)
    Tensor2D_set_index(t, r, col, val);
}

void
Tensor2D_fill_row ( Tensor2D *t, const size_t row, const double val )
{
  for (size_t c = 0; c < t->cols; ++c)
    Tensor2D_get_index(t, row, c, val);
}

/* get and set */
double
Tensor2D_get_index ( Tensor2D *t, const size_t row, const size_t col )
{
  return t->data[ row * t->cols + col ];
}

void
Tensor2D_set_index ( Tensor2D *t, const size_t row, const size_t col, const double val )
{
  t->data[ row * t->cols + col ] = val;
}

/* data->tensor work */
Tensor2D *
Tensor2D_load_xvalue_tensor ( double *xvalues, const size_t length )
{
  /* create 2D column vector */
  Tensor2D *t = Tensor2D_create(length, 2);

  /* set first column to 1s */
  Tensor2D_fill_column(t, 0, 1);

  /* copy values to column */
  for (size_t i = 0; i < length; ++i)
    Tensor2D_set_index(t, i, 1, yvalues[i]);

  return t;
}

Tensor2D *
Tensor2D_load_yvalue_tensor ( double *yvalues, const size_t length )
{
  /* create column vector */
  Tensor2D *t = Tensor2D_create(length, 1);

  /* copy values to column */
  for (size_t i = 0; i < length; ++i)
    Tensor2D_set_index(t, i, 0, yvalues[i]);

  return t;
}
