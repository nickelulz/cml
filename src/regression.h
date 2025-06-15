#ifndef REGRESSION_HEADER
#define REGRESSION_HEADER

typedef {
  double coefficient;
  double intercept;
  double r_squared;
} RegressionResult;

RegressionResult calculate_linear_regression ( const double *x, const double *y, const size_t size );

#endif
