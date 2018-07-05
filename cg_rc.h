#ifndef __CG_RC_H_
#define __CG_RC_H_

# include <cstdlib>
# include <iostream>
# include <iomanip>
# include <cmath>
# include <ctime>

int cg_rc ( int n, double b[], double x[], double r[], 
		double z[], double p[], double q[], int job );

//****************************************************************************80
//
//  Purpose:
//
//    R8MAT_MV multiplies a matrix times a vector.
//
//  Discussion:
//
//    An R8MAT is a doubly dimensioned array of R8 values, stored as a vector
//    in column-major order.
//
//    For this routine, the result is returned as an argument.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 April 2007
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int M, N, the number of rows and columns of the matrix.
//
//    Input, double A[M,N], the M by N matrix.
//
//    Input, double X[N], the vector to be multiplied by A.
//
//    Output, double AX[M], the product A*X.
//
void r8mat_mv ( int m, int n, double a[], double x[], double ax[] );

double *r8vec_uniform_01_new ( int n, int &seed );

void timestamp ( );

double *wathen ( int nx, int ny, int n );

int wathen_order ( int nx, int ny );

#endif
