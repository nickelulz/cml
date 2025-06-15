#ifndef TENSOR_HEADER
#define TENSOR_HEADER

/* simple linear algebra library */
typedef struct {
  double *data;
  size_t rows, cols;
} Tensor2D;

/* basic operations */
Tensor2D *Tensor2D_create  ( size_t rows, size_t cols );
Tensor2D *Tensor2D_copy    ( Tensor2D *t );
void      Tensor2D_destroy ( Tensor2D *t );

/* actual math operations */
Tensor2D *Tensor2D_transpose  ( Tensor2D *t );
Tensor2D *Tensor2D_mult       ( Tensor2D *a, Tensor2D *b );
Tensor2D *Tensor2D_sq_inverse ( Tensor2D *t );

/* data manipulation */
void Tensor2D_fill_column ( Tensor2D *t, const size_t col, const double val );
void Tensor2D_fill_row    ( Tensor2D *t, const size_t row, const double val );

/* get and set */
double Tensor2D_get_index ( Tensor2D *t, const size_t row, const size_t col );
void   Tensor2D_set_index ( Tensor2D *t, const size_t row, const size_t col, const double val );

/* data->tensor work */
Tensor2D *Tensor2D_load_xvalue_tensor ( double *xvalues, const size_t length );
Tensor2D *Tensor2D_load_yvalue_tensor ( double *yvalues, const size_t length );

#endif
