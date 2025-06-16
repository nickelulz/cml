#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <termios.h>
#include "gnuplot_i.h"

#include "tensor.h"
#include "regression.h"

static int
getch ( void ) 
{
  int ch;
  struct termios oldt, newt;
  
  tcgetattr ( STDIN_FILENO, &oldt );
  newt = oldt;
  newt.c_lflag &= ~( ICANON | ECHO );
  tcsetattr ( STDIN_FILENO, TCSANOW, &newt );
  ch = getchar();
  tcsetattr ( STDIN_FILENO, TCSANOW, &oldt );
  
  return ch;
}

int
main(int argc, char **argv)
{
  (void) argc;
  (void) argv;
  
  const size_t size    = 10; 
  double x[] = {1, 3, 4, 6,  7,  9,  11, 12, 14, 15};
  double y[] = {4, 7, 9, 12, 14, 18, 20, 24, 27, 29};
  
  RegressionResult result = calculate_linear_regression(x, y, size);
  
  gnuplot_ctrl *fig = gnuplot_init();

  /* set the x and y limits */
  gnuplot_cmd(fig, "set xrange [0:15]");
  gnuplot_cmd(fig, "set yrange [0:30]");
  
  gnuplot_set_xlabel(fig, "x");
  gnuplot_set_ylabel(fig, "f(x)");

  /* plot x and y */
  gnuplot_setstyle(fig, "points") ;
  gnuplot_plot_xy(fig, x, y, size, "data");
  
  char regression_label_buffer[1024];

  snprintf(regression_label_buffer, 1024, "f(x) = %.3fx + %.3f, R^2 = %.3f",
	   result.coefficient,
	   result.intercept,
	   result.r_squared);

  gnuplot_setstyle(fig, "lines");
  gnuplot_plot_slope(fig,
                     result.coefficient,
                     result.intercept,
                     regression_label_buffer);

  /* wait for user input */
  printf("Press any key to exit.");
  getch();
  printf("\n");

  /* end session */
  gnuplot_close(fig);
}
