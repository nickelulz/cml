#ifndef REGRESSION_HEADER
#define REGRESSION_HEADER

typedef {
  double coefficient, intercept, r_squared;
} RegressionResult;

RegressionResult calculate_linear_regression(const double *x, const double *y, const size_t size);

#endif
