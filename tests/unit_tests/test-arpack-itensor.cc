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
  double eps = 1.0e-08;
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
  ITensor U(Kp), D, V;
  svd(A, U, D, V, solver, {"Truncate", false});

  PrintData(U);

  EXPECT_TRUE(norm(A - U * D * V) < eps);
}

} // namespace itensor