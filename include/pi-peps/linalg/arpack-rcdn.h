#ifndef __ARPACK_RCI_H_
#define __ARPACK_RCI_H_

#include "pi-peps/config.h"
// #include "arpack.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <vector>
#include <complex>

#ifndef BLASINT
#  define BLASINT int
#endif

extern "C" void dnaupd_(int* ido,
                        char* bmat,
                        unsigned int* n,
                        char* which,
                        unsigned int* nev,
                        const double* tol,
                        double* resid,
                        int* ncv,
                        double* v,
                        int* ldv,
                        int* iparam,
                        int* ipntr,
                        double* workd,
                        double* workl,
                        int* lworkl,
                        int* info);

extern "C" void dneupd_(int* rvec,
                        char* howmany,
                        int* select,
                        double* d,
                        double* di,
                        double* z,
                        int* ldz,
                        double* sigmar,
                        double* sigmai,
                        double* workev,
                        char* bmat,
                        unsigned int* n,
                        char* which,
                        unsigned int* nev,
                        double* tol,
                        double* resid,
                        int* ncv,
                        double* v,
                        int* ldv,
                        int* iparam,
                        int* ipntr,
                        double* workd,
                        double* workl,
                        int* lworkl,
                        int* info);

template <class T>
struct ARDNS {
  T& mvp;

  ARDNS(T& mmvp) : mvp(mmvp) {}

  void real_nonsymm(BLASINT const N,
                           BLASINT const nev,
                           BLASINT const max_ncv,
                           double const tol,
                           BLASINT const maxIter,
                           std::vector< std::complex<double> > & ev,
                           std::vector<double> & V,
                           bool rvec = false,
                           bool dbg = false) {
    BLASINT ncv = std::max(max_ncv, 2 * nev + 1);
    BLASINT ldv = N;
    BLASINT lworkl = 3 * (ncv * ncv) + 6 * ncv;

    if (dbg) std::cout << "N " << N << "\n"
              << "nev " << nev << "\n"
              << "tol " << tol << "\n"
              << "maxIter " << maxIter << "\n"
              << "ncv " << ncv << "\n"
              << "ldv " << ldv << "\n"
              << "lworkl " << lworkl << "\n";

    // ***** Local Arrays **********************************************
    // dnaupd
    std::vector<double> resid(N);
    V.resize(ncv * N);
    std::vector<double> workd(3 * N, 0.0);
    std::vector<double> workl(lworkl, 0.0);
    std::array<BLASINT, 11> iparam{};
    std::array<BLASINT, 14> ipntr{};

    // ***** Local Scalars *********************************************
    iparam[0] = 1;        // use exact shifts
    iparam[2] = maxIter;  // maximal number of iterations
    iparam[3] = 1;
    iparam[4] = 0;  // number of ev found by arpack.
    iparam[6] = 1;  // mode 1 of dnaupd (???) related to use of exact shifts
                    // iparam[0] = 1

    BLASINT info = 0, ido = 0;

    char bmat[2] = "I";
    char which[3] = "LM";
    unsigned int n = N;
    unsigned int Nev = nev;
    double Tol = tol;
    while (ido != 99) {
      // arpack::saupd(ido, arpack::bmat::identity, N,
      //               arpack::which::largest_magnitude, nev, tol, resid.data(),
      //               ncv, V.data(), ldv, iparam.data(), ipntr.data(),
      //               workd.data(), workl.data(), lworkl, info);

      dnaupd_(&ido,
              bmat,   // solve Ax = lambda Bx where B = Identity
              &n,     // linear size of A
              which,  // solve for largest eigenvalues
              &Nev, &tol, resid.data(), &ncv, V.data(), &ldv, iparam.data(),
              ipntr.data(), workd.data(), workl.data(), &lworkl, &info);

      // perform matrix vector multiplication Ax -> y
      mvp(&(workd[ipntr[0] - 1]),  // = x
          &(workd[ipntr[1] - 1])   // = y
      );
    }

    // check number of ev found by arpack.
    if (iparam[4] != nev || info != 0) {
      std::cout << "[arpack::dnaupd] iparam[4] = " << iparam[4] << std::endl;
      std::cout << "[arpack::dnaupd] info=" << info << std::endl;
      // throw std::domain_error("Error inside ARPACK routines");
    } else {
      if (dbg) std::cout << "Processing result of [arpack::dnaupd]" << std::endl;
    }

    // ***** dnaupd exited succesfully, post-process result ************
    BLASINT ldz = N + 1;

    std::vector<double> dr(nev + 1);
    std::vector<double> di(nev + 1);
    std::vector<double> workev(3 * ncv);
    std::vector<double> z((N + 1) * (nev + 1));

    std::vector<int> select(ncv);
    // bool const rvec = false; // Extract eigenvectors Y/N ?
    int irvec = (rvec) ? 1 : 0;  // Extract eigenvectors Y/N ?

    double sigmar = 0.0;
    double sigmai = 0.0;

    char howmny[2] = "A";  // compute NEV Ritz vectors
    dneupd_(&irvec, howmny, select.data(),
            dr.data(),  // first column of d contains real part of eigenvalues
            di.data(),  // second column of d contains imag part of eigenvalues
            z.data(), &ldz,
            &sigmar,  //
            &sigmai,  //
            workev.data(),
            bmat,  // process for B = Identity
            &n,
            which,  // process for largest eigenvalues
            &Nev, &Tol, resid.data(), &ncv, V.data(), &ldv, iparam.data(),
            ipntr.data(), workd.data(), workl.data(), &lworkl, &info);

    if (info != 0) {
      std::cout << "[arpack::dneupd] info=" << info << std::endl;
      throw std::domain_error("[arpack::dneupd] Error in dneupd");
    } else {
      if (dbg) std::cout << "[arpack::dneupd] Processing done" << std::endl;
    }

    // record eigenvalues
    ev.resize(nev);
    for (int i = 0; i < nev; ++i) ev[i] = (dr[i], di[i]);
  }

};

#endif
