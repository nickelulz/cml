#ifndef REGRESSION_HEADER
#define REGRESSION_HEADER

#include <stddef.h>

typedef struct {
  double coefficient;
  double intercept;
  double r_squared;
} RegressionResult;

RegressionResult calculate_linear_regression ( double *x, double *y, const size_t size );

#endif
