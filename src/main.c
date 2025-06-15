#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "gnuplot_i.h"

#include "tensor.h"
#include "regression.h"

int main(int argc, char **argv)
{
  (void) argc;
  (void) argv;
  
  const size_t size    = 10; 
  const double x[size] = {1, 3, 4, 6,  7,  9,  11, 12, 14, 15};
  const double y[size] = {4, 7, 9, 12, 14, 18, 20, 24, 27, 29};
  
  RegressionResult result = calculate_linear_regression(x, y, size);
  
  gnuplot_ctrl *fig = gnuplot_init();
  gnuplot_setstyle(fig, "lines");
  gnuplot_set_xlabel(h, "x");
  gnuplot_set_ylabel(h, "f(x)");

  char regression_label_buffer[1024];
  snprintf(regression_label_buffer, "f(x) = %.3fx + %.3f, R^2 = %.3f",
	   result.coefficient,
	   result.intercept,
	   result.r_squared);
  gnuplot_plot_slope(fig, 1.0, 0.0, regression_label_buffer);
  sleep(10000);
}
