#include "pi-peps/config.h"
#include <gtest/gtest.h>
#include "pi-peps/linalg/arpack-rcdn.h"
DISABLE_WARNINGS
#include "itensor/all.h"
ENABLE_WARNINGS

struct DiagMVP {
  int id;

  DiagMVP(int iid) : id(iid) {}

  void operator()(double const* const x, double* const y) {
    for (int i = 0; i < 1000; ++i) {
      y[i] = static_cast<double>(i + 1) * x[i];
    }
  }
};

namespace itensor {

  // matrix-vector multiplication by 10x10 diag matrix with entries (10,9,...)
  struct DiagMVP_itensor {
    int N;

    DiagMVP_itensor(int nn) : N(nn) {}

    void operator()(double const* const x, double* const y) {
      auto i = Index("i", N);
      auto ip = prime(i);

      // copy x
      std::vector<double> cpx(N);
      std::copy(x, x + N, cpx.data());

      // auto vecRefX = makeVecRef(cpx.data(),cpx.size());

      auto isX = IndexSet(i);
      auto X = ITensor(isX, Dense<double>(std::move(cpx)));
      // auto Y = ITensor({ip},Dense<double>(std::move(y)));

      auto A = ITensor(ip, i);
      for (int j = 1; j <= N; j++)
        A.set(ip(j), i(j), 10.0 - (j - 1));

      auto Y = A * X;
      Y.scaleTo(1.0);

      auto extractReal = [](Dense<Real> const& d) { return d.store; };

      auto yData = applyFunc(extractReal, Y.store());
      std::copy(yData.data(), yData.data() + N, y);
    }
  };

TEST(ArpackReal0, Default_cotr) {
  double eps = 1.0e-08;

  itensor::DiagMVP_itensor dmvp(10);
  ARDNS<itensor::DiagMVP_itensor> ardns(dmvp);

  std::vector<std::complex<double>> ev;
  std::vector<double> V;
  ardns.real_nonsymm(10, 2, 5, 0.0, 10 * 10, ev, V);

  // eigenvalues are sorted in ascending order
  EXPECT_TRUE((std::abs(ev[0]) - 9.0) < eps);
  EXPECT_TRUE((std::abs(ev[1]) - 10.0) < eps);
}

TEST(ArpackRealSvd0, Default_cotr) {
  double eps = 1.0e-07;

  Index K = Index("k", 5);
  Index Kp = prime(K);
  ITensor A(K, Kp);
  A.set(K(1), Kp(1), 6.80);
  A.set(K(1), Kp(2), -6.05);
  A.set(K(1), Kp(3), -0.45);
  A.set(K(1), Kp(4), 8.32);
  A.set(K(1), Kp(5), -9.67);
  A.set(K(2), Kp(1), -2.11);
  A.set(K(2), Kp(2), -3.30);
  A.set(K(2), Kp(3), 2.58);
  A.set(K(2), Kp(4), 2.71);
  A.set(K(2), Kp(5), -5.14);
  A.set(K(3), Kp(1), 5.66);
  A.set(K(3), Kp(2), 5.36);
  A.set(K(3), Kp(3), -2.70);
  A.set(K(3), Kp(4), 4.35);
  A.set(K(3), Kp(5), -7.26);
  A.set(K(4), Kp(1), 5.97);
  A.set(K(4), Kp(2), -4.44);
  A.set(K(4), Kp(3), 0.27);
  A.set(K(4), Kp(4), -7.17);
  A.set(K(4), Kp(5), 6.08);
  A.set(K(5), Kp(1), 8.23);
  A.set(K(5), Kp(2), 1.08);
  A.set(K(5), Kp(3), 9.04);
  A.set(K(5), Kp(4), 2.14);
  A.set(K(5), Kp(5), -6.87);

  ArpackSvdSolver solver = ArpackSvdSolver();
  ITensor U(Kp), D, Vt;
  svd(A, U, D, Vt, solver, {"Maxm", 4});

  PrintData(U);
  PrintData(D);
  PrintData(Vt);

  // true singular values of A = {21.5538, 12.5376, 9.5126, 8.4900, 1.759784917702199}
  double n = norm(A - U * D * Vt);
  EXPECT_TRUE(std::sqrt( std::abs(n*n - 1.759784917702199*1.759784917702199) ) < eps);
}

TEST(ArpackRealSvd1, Default_cotr) {
  double eps = 1.0e-07;

  Index a("a", 3), b("b", 2);
  ITensor A(a, b);
  A.set(a(1), b(1), 1);
  A.set(a(2), b(1), 2);
  A.set(a(3), b(1), 3);
  A.set(a(1), b(2), 4);
  A.set(a(2), b(2), 5);
  A.set(a(3), b(2), 6);

  ArpackSvdSolver solver = ArpackSvdSolver();
  ITensor U(b), D, Vt;
  svd(A, U, D, Vt, solver, {"Maxm", 1});

  PrintData(U);
  PrintData(D);
  PrintData(Vt);

  // true singular values of A = {9.5080, 0.772869635673485}
  double n = norm(A - U * D * Vt);
  EXPECT_TRUE( std::sqrt( std::abs(n*n - 0.772869635673485*0.772869635673485) ) < eps);
}

TEST(ArpackRealSvd2, Default_cotr) {
  double eps = 1.0e-07;

  Index a("a", 2), b("b", 3);
  ITensor A(a, b);
  A.set(a(1), b(1), 1);
  A.set(a(2), b(1), 4);
  A.set(a(1), b(2), 2);
  A.set(a(2), b(2), 5);
  A.set(a(1), b(3), 3);
  A.set(a(2), b(3), 6);

  ArpackSvdSolver solver = ArpackSvdSolver();
  ITensor U(b), D, Vt;
  svd(A, U, D, Vt, solver, {"Maxm", 1});

  PrintData(U);
  PrintData(D);
  PrintData(Vt);

  // true singular values of A = {9.5080, 0.772869635673485}
  double n = norm(A - U * D * Vt);
  EXPECT_TRUE( std::sqrt( std::abs(n*n - 0.772869635673485*0.772869635673485) ) < eps);
}

} // namespace itensor
