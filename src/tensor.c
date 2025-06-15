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

Tensor2D *
Tensor2D_transpose ( Tensor2D *t )
{
  Tensor2D *new = Tensor2D_create(t->cols, t->rows);
  for (size_t r = 0; r < t->rows; ++r)
    for (size_t c = 0; c < t->cols; ++c)
      Tensor2D_set_index(new, c, r, Tensor2D_get_index(t, r, c));
  return new;
}
  
void
Tensor2D_destroy ( Tensor2D *t )
{
  free(t->data);
  free(t);
  &t = NULL;
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
