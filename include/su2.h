#ifndef __SU2_
#define __SU2_

#include <algorithm>
#include <complex>
#include <cmath>
#include "itensor/all.h"

// Supported types of 1-site operators
enum SU2O {
    SU2_Id,     // Identity
    SU2_S_Z,    // Projection on S_z
    SU2_S_Z2,   // S_z^2
    SU2_S_P,    // S_plus
    SU2_S_M     // S_minus
};

/*
 * Return spin operator with indices s, prime(s)
 *
 */
itensor::ITensor SU2_getSpinOp(SU2O su2o, itensor::Index const& s,
	bool DBG = false);

double SU2_getCG(int j1, int j2, int j, int m1, int m2, int m);

int Factorial(int x);

#endif