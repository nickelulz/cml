#include "regression.h"
#include "tensor.h"

RegressionResult
calculate_linear_regression ( const double *x, const double *y, const size_t size )
{
  Tensor2D *x_tensor = Tensor2D_load_xvalue_tensor ( x, size );
  Tensor2D *y_tensor = Tensor2D_load_yvalue_tensor ( y, size );
}
