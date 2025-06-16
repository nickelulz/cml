#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "tensor.h"

/* basic operations */
void
Tensor2D_print ( Tensor2D *t )
{
  if ( t == NULL ) {
    fprintf(stderr, "cannot print NULL tensor\n");
    return;
  }

  // Iterate through rows and columns of the tensor
  printf("TENSOR (%zu x %zu):\n", t->rows, t->cols);
  printf("[");
  for (size_t i = 0; i < t->rows; ++i) {
    for (size_t j = 0; j < t->cols; ++j) {
      // Print each element with a space separating them
      if (i > 0 && j == 0)
        printf(" ");
      printf(" %*.*f ", 5, 1, Tensor2D_get_index(t, i, j));
    }
    // Print a newline at the end of each row
    if (i < t->rows - 1)
      printf("\n");
    else
      printf("]\n\n");
  }
}

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
  memcpy(new->data, t->data, sizeof(double) * t->rows * t->cols);
  return new;
}
  
void
Tensor2D_destroy ( Tensor2D **t )
{
  if (t && *t) {
    free((*t)->data);
    free(*t);
    *t = NULL;
  }
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
  Tensor2D_print(a);
  Tensor2D_print(b);
  
  if (a->cols != b->rows) {
    fprintf(stderr,
	    "invalid MULT call due to mismatched size: "
	    "a (%zu cols) != b (%zu rows)\n",
	    a->cols, b->rows);
    return NULL;
  }
  
  Tensor2D *result = Tensor2D_create ( a->rows, b->cols );
  
  fprintf(stderr, "beginning mult..\n");
  
  for (size_t i = 0; i < a->rows; ++i)
    for (size_t j = 0; j < b->cols; ++j) {
      double sum = 0.0;
      for (size_t k = 0; k < a->cols; ++k)
        sum += Tensor2D_get_index(a, i, k) * Tensor2D_get_index(b, k, j);
      Tensor2D_set_index(result, i, j, sum);
    }
  
  fprintf(stderr, "mult complete\n");

  return result;
}

static void
Tensor2D_inplace_multiply_row ( Tensor2D *t, size_t row, double scalar )
{
  for (size_t col = 0; col < t->cols; ++col) {
    double prev = Tensor2D_get_index( t, row, col );
    Tensor2D_set_index( t, row, col, prev * scalar );
  }
}
  
static void
Tensor2D_inplace_divide_row ( Tensor2D *t, size_t row, double scalar )
{
  Tensor2D_inplace_multiply_row ( t, row, 1 / scalar );
}

static void
Tensor2D_inplace_swap_rows ( Tensor2D *t, size_t row_a, size_t row_b )
{
  /* make row buffer */
  double *row_a_buf = (double*) malloc(sizeof(double) * t->cols);
  
  /* copy row a to the buffer */
  for (size_t c = 0; c < t->cols; ++c)
    row_a_buf[c] = Tensor2D_get_index( t, row_a, c );
  
  /* overwrite row a with row b */
  for (size_t c = 0; c < t->cols; ++c)
    Tensor2D_set_index( t, row_a, c,
			Tensor2D_get_index( t, row_b, c ) );

  /* overwrite row b from the buffer */
  for (size_t c = 0; c < t->cols; ++c)
    Tensor2D_set_index( t, row_b, c, row_a_buf[c] );

  /* free the buffer */
  free(row_a_buf);
}

/* TODO: fix the fact that the intermediate augment isnt modified, but the OG matrix is
         by producing a copy of the OG at the beginning OR writing its values to the ia
         then performing all operations DIRECTLY onto the ia inplace. */
Tensor2D *
Tensor2D_sq_inverse ( Tensor2D *t )
{
  /* tensor must be square */
  if (t->rows != t->cols)
    return NULL;

  /* copy t to a new tensor */
  t = Tensor2D_copy( t );
  
  /* gauss-jordan elimination to calculate the inverse
     https://en.wikipedia.org/wiki/Gaussian_elimination */
  Tensor2D *intermediate_augment = Tensor2D_create( t->rows, t->cols + t->rows );
  Tensor2D *ia = intermediate_augment;

  /* copy the data to intermediate augment */
  for ( size_t r = 0; r < t->rows; ++r ) {
    for ( size_t c = 0; c < t->cols; ++c )
      Tensor2D_set_index( ia, r, c, Tensor2D_get_index( t, r, c ) );

    /* set up the identity matrix */
    Tensor2D_set_index( ia, r, r + t->rows, 1 );
  }

  fprintf(stderr, "INTERMEDIATE AUGMENT TENSOR BEFORE\n");
  Tensor2D_print( intermediate_augment );
    
  /* forward elimination */
  for ( size_t pivot = 0; pivot < t->rows; ++pivot ) {
    /* find the pivot row (with the largest abs value in the column) */
    size_t max_row = pivot;
    double max_val = fabs( Tensor2D_get_index( ia, pivot, pivot ) );

    for ( size_t r = pivot + 1; r < t->rows; ++r ) {
      double val = fabs( Tensor2D_get_index( ia, r, pivot ) );
      if ( val > max_val ) {
        max_val = val;
        max_row = r;
      }
    }

    /* matrix is singular */
    if (max_val == 0) {
      Tensor2D_destroy( &t );
      Tensor2D_destroy( &ia );
      return NULL;
    }

    /* swap pivot row */
    Tensor2D_inplace_swap_rows( ia, pivot, max_row );

    /* normalize pivot row */
    Tensor2D_inplace_divide_row( ia, pivot, Tensor2D_get_index( ia, pivot, pivot ) );

    /* eliminate other rows */
    for (size_t r = 0; r < t->rows; ++r) {
      if (r == pivot)
        continue;

      double factor = Tensor2D_get_index( ia, r, pivot );
      for (size_t c = 0; c < ia->cols; ++c)
        Tensor2D_set_index( ia, r, c,
                            Tensor2D_get_index( ia, r, c ) -
                            factor * Tensor2D_get_index( ia, pivot, c ) );
    }
  }

  fprintf(stderr, "INTERMEDIATE AUGMENT TENSOR AFTER\n");
  Tensor2D_print( intermediate_augment );
  
  /* copy data */
  Tensor2D *sq_inv_result = Tensor2D_create ( t->rows, t->cols );
  for ( size_t r = 0; r < t->rows; ++r )
    for ( size_t c = 0; c < t->cols; ++c )
      Tensor2D_set_index ( sq_inv_result, r, c,
                           Tensor2D_get_index ( intermediate_augment, r, c + t->rows ) );
  
  /* free intermediate and return */
  Tensor2D_destroy( &t );
  Tensor2D_destroy( &intermediate_augment );
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
    Tensor2D_set_index(t, row, c, val);
}

/* get and set */
double
Tensor2D_get_index ( Tensor2D *t, const size_t row, const size_t col )
{
  if ( row >= t->rows || col >= t->cols ) {
    fprintf(stderr, "access GET error on tensor of size %zu x %zu at index (%zu, %zu)\n",
	    t->rows, t->cols, row, col);
    return DBL_MAX; /* return junk data */
  }
  
  return t->data[ row * t->cols + col ];
}

void
Tensor2D_set_index ( Tensor2D *t, const size_t row, const size_t col, const double val )
{
  if ( row >= t->rows || col >= t->cols ) {
    fprintf(stderr,
	    "access SET error on tensor of size %zu x %zu "
	    "at index (%zu, %zu) with value %.3f\n",
	    t->rows, t->cols, row, col, val);
    /* do nothing and return */
    return;
  }
  
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
    Tensor2D_set_index(t, i, 1, xvalues[i]);

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
