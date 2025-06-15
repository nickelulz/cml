#include "regression.h"
#include "tensor.h"

static double
sqr ( const double x )
{
  return x * x;
}

RegressionResult
calculate_linear_regression ( const double *x, const double *y, const size_t size )
{
  Tensor2D *x_tensor = Tensor2D_load_xvalue_tensor ( x, size );
  Tensor2D *y_tensor = Tensor2D_load_yvalue_tensor ( y, size );

  /* intermediate steps
     https://en.wikipedia.org/wiki/Linear_least_squares#Fitting_a_line */
  Tensor2D *x_tensor_transposed = Tensor2D_transpose( x_tensor );
  Tensor2D *x_gram = Tensor2D_mult( x_tensor, x_tensor_transposed );
  Tensor2D *x_gram_inverse = Tensor2D_sq_inverse( x_gram );
  Tensor2D *gram_inv_trans_prod = Tensor2D_mult( x_gram_inverse, x_tensor_transposed );
  Tensor2D *beta = Tensor2D_mult( gram_inv_trans_prod, y_tensor );

  /* free original tensors */
  Tensor2D_destroy( x_tensor );
  Tensor2D_destroy( y_tensor );
  
  /* free intermediate steps */
  Tensor2D_destroy( x_tensor_transposed );
  Tensor2D_destroy( x_gram );
  Tensor2D_destroy( x_gram_inverse );
  Tensor2D_destroy( gram_inv_trans_prod );

  /* obtain slope and intercept from beta */
  double intercept = Tensor2D_get_index( beta, 0, 0 );
  double slope     = Tensor2D_get_index( beta, 1, 0 );
  
  /* free beta */
  Tensor2D_destroy( beta );
  
  /* calculate residuals and R^2 */
  double predictions[size], residuals[size];
  double residual_sq_sum, total_sq_sum, mean_y_value = 0;
  
  for (size_t i = 0; i < size; ++i)
    mean_y += y[i];
  mean_y = mean_y / size;

  for (size_t i = 0; i < size; ++i) {
    predictions[i] = intercept + slope * x[i];
    residuals[i] = y[i] - predictions[i];

    residual_sq_sum += sqr( residuals[i] );
    total_sq_sum += sqr( y[i] - mean_y_value );
  }

  double r_squared = 1 - residual_sq_sum / total_sq_sum;

  /* save data and output */
  return (RegressionResult) {
    .coefficient = slope;
    .intercept = intercept;
    .r_squared = r_squared;
  };
}
