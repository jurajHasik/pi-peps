#ifndef __LATTICE_H_
#define __LATTICE_H_

#include "pi-peps/config.h"
#include <array>
#include <iostream>

// namespace itensor {

/*
 * Generic displacement vector on a square lattice.
 * Support +,-,-=,+=,==,!=
 */
struct Shift {
  std::array<int, 2> d;

  Shift() : d({0, 0}) {}

  Shift(int dx, int dy) : d({dx, dy}) {}

  Shift Shift_LEFT() { return Shift(-1, 0); }
  Shift Shift_UP() { return Shift(0, -1); }
  Shift Shift_RIGHT() { return Shift(1, 0); }
  Shift Shift_DOWN() { return Shift(0, 1); }

  bool operator==(const Shift& s) const {
    return (this->d[0] == s.d[0]) && (this->d[1] == s.d[1]);
  }

  bool operator!=(const Shift& s) const {
    return (this->d[0] != s.d[0]) || (this->d[1] != s.d[1]);
  }

  Shift operator*(int x) const { return Shift(x * this->d[0], x * this->d[1]); }

  Shift operator+(Shift const& s) const {
    return Shift(this->d[0] + s.d[0], this->d[1] + s.d[1]);
  }

  Shift operator-(Shift const& s) const {
    return Shift(this->d[0] - s.d[0], this->d[1] - s.d[1]);
  }

  Shift& operator+=(Shift const& s) {
    this->d[0] += s.d[0];
    this->d[1] += s.d[1];
    return *this;
  }

  Shift& operator-=(Shift const& s) {
    this->d[0] -= s.d[0];
    this->d[1] -= s.d[1];
    return *this;
  }
};

Shift operator*(int x, Shift const& s);

int dirFromShift(Shift const& s);

std::ostream& operator<<(std::ostream& os, Shift const& s);

/*
 * A vertex of a square lattice. One can obtain new vertices
 * by applying displacement: Vertex <- Vertex [+,-,+=,-=] Shift
 */
struct Vertex {
  std::array<int, 2> r;

  Vertex() : r({0, 0}) { /* Default Vertex */
  }

  Vertex(int x, int y) : r({x, y}) {}

  bool operator==(const Vertex& v) const {
    return (this->r[0] == v.r[0]) && (this->r[1] == v.r[1]);
  }

  bool operator!=(const Vertex& v) const {
    return (this->r[0] != v.r[0]) || (this->r[1] != v.r[1]);
  }

  bool operator>(const Vertex& v) const { return this->r > v.r; }

  bool operator<(const Vertex& v) const { return this->r < v.r; }

  Vertex operator+(Shift const& s) const {
    return Vertex(this->r[0] + s.d[0], this->r[1] + s.d[1]);
  }

  Vertex operator-(Shift const& s) const {
    return Vertex(this->r[0] - s.d[0], this->r[1] - s.d[1]);
  }

  Vertex& operator+=(Shift const& s) {
    this->r[0] += s.d[0];
    this->r[1] += s.d[1];
    return *this;
  }

  Vertex& operator-=(Shift const& s) {
    this->r[0] -= s.d[0];
    this->r[1] -= s.d[1];
    return *this;
  }
};

std::ostream& operator<<(std::ostream& os, Vertex const& v);

// }

#endif
