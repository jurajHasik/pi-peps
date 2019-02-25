#include "p-ipeps/config.h"
#include "p-ipeps/lattice.h"

// namespace itensor { 

Shift operator * (int x, Shift const& s) {
  return s * x;
}

int dirFromShift(Shift const& s) {
    if (s == Shift(-1,0)) return 0; // LEFT
    if (s == Shift(1,0)) return 2; // RIGHT
    if (s == Shift(0,1)) return 3; // DOWN
    if (s == Shift(0,-1)) return 1; // UP

    throw std::runtime_error("[dirFromShift] Invalid Shift");

    return -1;
}

// }
