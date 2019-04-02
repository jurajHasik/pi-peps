#ifndef __ARPACK_RCI_H_
#define __ARPACK_RCI_H_

#include "pi-peps/config.h"
#ifdef PEPS_WITH_ARPACK

#  include "pi-peps/linalg/itensor-svd-solvers.h"

#  include <array>
#  include <cmath>
#  include <complex>
#  include <iostream>
#  include <vector>

#  ifndef BLASINT
#    define BLASINT int
#  endif

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

extern "C" void dsaupd_(int* ido,
                        char const* bmat,
                        int* n,
                        char const* which,
                        int* nev,
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

extern "C" void dseupd_(int* rvec,
                        char const* howmny,
                        int const* select,
                        double* d,
                        double* z,
                        int* ldz,
                        double* sigma,
                        char const* bmat,
                        int* n,
                        char const* which,
                        int* nev,
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
                    std::vector<std::complex<double>>& ev,
                    std::vector<double>& V,
                    bool rvec = false,
                    bool dbg = false) {
    BLASINT ncv = std::max(max_ncv, 2 * nev + 1);
    BLASINT ldv = N;
    BLASINT lworkl = 3 * (ncv * ncv) + 6 * ncv;

    if (dbg)
      std::cout << "N " << N << "\n"
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
      if (dbg)
        std::cout << "Processing result of [arpack::dnaupd]" << std::endl;
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
      if (dbg)
        std::cout << "[arpack::dneupd] Processing done" << std::endl;
    }

    // record eigenvalues
    ev.resize(nev);
    for (int i = 0; i < nev; ++i) {
      ev[i] = std::complex<double>(dr[i], di[i]);
    }
  }
};

namespace itensor {

  struct ArpackSvdSolver : SvdSolver {
    void solve(MatRefc<Real> const& M,
               MatRef<Real> const& U,
               VectorRef const& D,
               MatRef<Real> const& V,
               Args const& args) {
      bool dbg = args.getBool("svd_dbg", false);

      auto Mr = nrows(M);
      auto Mc = ncols(M);
      if (Mr < Mc) {
        if (dbg)
          std::cout << "[ArpackSvdSolver::solve] transposing " << Mr << "x"
                    << Mc << " -> " << Mc << "x" << Mr << std::endl;
        solve(transpose(M), V, D, U, args);
        conjugate(V);
        conjugate(U);
        return;
      }

#  ifdef DEBUG
      if (!(nrows(U) == Mr && ncols(U) == Mr))
        throw std::runtime_error("SVD (ref version), wrong size of U");
      if (!(nrows(V) == Mc && ncols(V) == Mr))
        throw std::runtime_error("SVD (ref version), wrong size of V");
      if (D.size() != Mc)
        throw std::runtime_error("SVD (ref version), wrong size of D");
#  endif

      // https://github.com/opencollab/arpack-ng/blob/master/EXAMPLES/SVD/dsvd.f
      //-----------------------------------------------------------------------
      //
      //     %------------------------------------------------------%
      //     | Storage Declarations:                                |
      //     |                                                      |
      //     | It is assumed that A is M by N with M .ge. N.        |
      //     |                                                      |
      //     | The maximum dimensions for all arrays are            |
      //     | set here to accommodate a problem size of            |
      //     | M .le. MAXM  and  N .le. MAXN                        |
      //     |                                                      |
      //     | The NEV right singular vectors will be computed in   |
      //     | the N by NCV array V.                                |
      //     |                                                      |
      //     | The NEV left singular vectors will be computed in    |
      //     | the M by NEV array U.                                |
      //     |                                                      |
      //     | NEV is the number of singular values requested.      |
      //     |     See specifications for ARPACK usage below.       |
      //     |                                                      |
      //     | NCV is the largest number of basis vectors that will |
      //     |     be used in the Implicitly Restarted Arnoldi      |
      //     |     Process.  Work per major iteration is            |
      //     |     proportional to N*NCV*NCV.                       |
      //     |                                                      |
      //     | You must set:                                        |
      //     |                                                      |
      //     | MAXM:   Maximum number of rows of the A allowed.     |
      //     | MAXN:   Maximum number of columns of the A allowed.  |
      //     | MAXNEV: Maximum NEV allowed                          |
      //     | MAXNCV: Maximum NCV allowed                          |
      //    %------------------------------------------------------%
      //

      int m = Mr;
      int n = Mc;
      int requested_sv = args.getInt("Maxm", Mc);
      if (requested_sv > n - 1) {
        throw std::runtime_error(
          "[ArpackSvdSolver::solve] Number of singular values requested >= "
          "min(nrows,ncols)");
      }
      int nev = requested_sv;
      int requested_ncv = args.getInt("arpack_svd_ncv", 2 * nev + nev / 2);
      int ncv = std::min(requested_ncv, n);
      int ldv = n;
      int ldu = m;
      //
      //     %--------------%
      //     | Local Arrays |
      //     %--------------%
      //

      std::vector<double> v(ldv * ncv, 0.0);
      auto u = U.data();
      std::vector<double> workl(ncv * (ncv + 8), 0.0);
      std::vector<double> workd(3 * n, 0.0);
      std::vector<double> s(2 * ncv, 0.0);
      std::vector<double> resid(n, 0.0);
      std::vector<double> ax(m, 0.0);

      std::vector<int> select(ncv, 0);
      std::vector<int> iparam(11, 0);
      std::vector<int> ipntr(11, 0);

      // c
      // c     %---------------%
      // c     | Local Scalars |
      // c     %---------------%
      // c

      std::string bmat;
      std::string which;
      int ido, lworkl, info, ierr, ishfts, maxitr, mode1, nconv;
      int rvec;
      double tol = args.getReal("arpack_svd_tol", 0.0);
      double sigma, temp;

      // c
      // c     %-----------------------------%
      // c     | BLAS & LAPACK routines used |
      // c     %-----------------------------%
      // c
      //       Double precision
      //      &                 dnrm2
      //       external         dnrm2, daxpy, dcopy, dscal
      // c
      // c     %-----------------------%
      // c     | Executable Statements |
      // c     %-----------------------%
      // c
      // c     %-------------------------------------------------%
      // c     | The following include statement and assignments |
      // c     | initiate trace output from the internal         |
      // c     | actions of ARPACK.  See debug.doc in the        |
      // c     | DOCUMENTS directory for usage.  Initially, the  |
      // c     | most useful information will be a breakdown of  |
      // c     | time spent in the various stages of computation |
      // c     | given by setting msaupd = 1.                    |
      // c     %-------------------------------------------------%
      // c
      //       include 'debug.h'
      //       ndigit = -3
      //       logfil = 6
      //       msgets = 0
      //       msaitr = 0
      //       msapps = 0
      //       msaupd = 1
      //       msaup2 =
      //       mseigt = 0
      //       mseupd = 0
      // c
      // c     %------------------------------------------------%
      // c     | Specifications for ARPACK usage are set        |
      // c     | below:                                         |
      // c     |                                                |
      // c     |    1) NEV = sets number of singular values     |
      //       |       to be computed                           |
      // c     |                                                |
      // c     |    2) NCV sets the length of the Arnoldi       |
      // c     |       factorization                            |
      // c     |                                                |
      // c     |    3) This is a standard problem               |
      // c     |         (indicated by bmat  = 'I')             |
      // c     |                                                |
      // c     |    4) Ask for the NEV singular values of       |
      // c     |       largest magnitude                        |
      // c     |         (indicated by which = 'LM')            |
      // c     |       See documentation in DSAUPD for the      |
      // c     |       other options SM, BE.                    |
      // c     |                                                |
      // c     | Note: NEV and NCV must satisfy the following   |
      // c     |       conditions:                              |
      // c     |                 NEV <= MAXNEV,                 |
      // c     |             NEV + 1 <= NCV <= MAXNCV           |
      // c     %------------------------------------------------%
      // c
      bmat = "I";
      which = "LM";
      // c
      // c     %-----------------------------------------------------%
      // c     | Specification of stopping rules and initial         |
      // c     | conditions before calling DSAUPD                    |
      // c     |                                                     |
      // c     |           abs(sigmaC - sigmaT) < TOL*abs(sigmaC)    |
      // c     |               computed   true                       |
      // c     |                                                     |
      // c     |      If TOL .le. 0,  then TOL <- macheps            |
      // c     |              (machine precision) is used.           |
      // c     |                                                     |
      // c     | IDO  is the REVERSE COMMUNICATION parameter         |
      // c     |      used to specify actions to be taken on return  |
      // c     |      from DSAUPD. (See usage below.)                |
      // c     |                                                     |
      // c     |      It MUST initially be set to 0 before the first |
      // c     |      call to DSAUPD.                                |
      // c     |                                                     |
      // c     | INFO on entry specifies starting vector information |
      // c     |      and on return indicates error codes            |
      // c     |                                                     |
      // c     |      Initially, setting INFO=0 indicates that a     |
      // c     |      random starting vector is requested to         |
      // c     |      start the ARNOLDI iteration.  Setting INFO to  |
      // c     |      a nonzero value on the initial call is used    |
      // c     |      if you want to specify your own starting       |
      // c     |      vector (This vector must be placed in RESID.)  |
      // c     |                                                     |
      // c     | The work array WORKL is used in DSAUPD as           |
      // c     | workspace.  Its dimension LWORKL is set as          |
      // c     | illustrated below.                                  |
      // c     %-----------------------------------------------------%
      // c
      lworkl = ncv * (ncv + 8);
      info = 0;
      ido = 0;
      // c
      // c     %---------------------------------------------------%
      // c     | Specification of Algorithm Mode:                  |
      // c     |                                                   |
      // c     | This program uses the exact shift strategy        |
      // c     | (indicated by setting IPARAM(1) = 1.)             |
      // c     | IPARAM(3) specifies the maximum number of Arnoldi |
      // c     | iterations allowed.  Mode 1 of DSAUPD is used     |
      // c     | (IPARAM(7) = 1). All these options can be changed |
      // c     | by the user. For details see the documentation in |
      // c     | DSAUPD.                                           |
      // c     %---------------------------------------------------%
      // c
      ishfts = 1;
      maxitr = n;
      mode1 = 1;
      // c
      iparam[0] = ishfts;
      iparam[2] = maxitr;
      iparam[6] = mode1;
      // c
      // c     %------------------------------------------------%
      // c     | M A I N   L O O P (Reverse communication loop) |
      // c     %------------------------------------------------%
      // c
      // c        %---------------------------------------------%
      // c        | Repeatedly call the routine DSAUPD and take |
      // c        | actions indicated by parameter IDO until    |
      // c        | either convergence is indicated or maxitr   |
      // c        | has been exceeded.                          |
      // c        %---------------------------------------------%
      // c
      // av(n,x,y) computes y<-A*x
      // atv(n,y,w) computes w<-At*y
      auto av = [&M, &Mr, &Mc](double* x, double* y) {
        auto x_vec = makeVecRef(x, Mc);
        auto y_vec = makeVecRef(y, Mr);
        mult(M, x_vec, y_vec);
      };

      auto avt = [&M, &Mr, &Mc](double* x, double* y) {
        auto x_vec = makeVecRef(x, Mr);
        auto y_vec = makeVecRef(y, Mc);
        mult(M, x_vec, y_vec, true);  // transpose M
      };

      do {
        dsaupd_(&ido, bmat.c_str(), &n, which.c_str(), &nev, &tol, resid.data(),
                &ncv, v.data(), &ldv, iparam.data(), ipntr.data(), workd.data(),
                workl.data(), &lworkl, &info);
        // c
        // c           %---------------------------------------%
        // c           | Perform matrix vector multiplications |
        // c           |              w <--- A*x       (av())  |
        // c           |              y <--- A'*w      (atv()) |
        // c           | The user should supply his/her own    |
        // c           | matrix vector multiplication routines |
        // c           | here that takes workd(ipntr(1)) as    |
        // c           | the input, and returns the result in  |
        // c           | workd(ipntr(2)).                      |
        // c           %---------------------------------------%
        // c
        // call av (m, n, workd(ipntr(1)), ax)
        av(&workd[ipntr[0] - 1], ax.data());
        // call atv (m, n, ax, workd(ipntr(2)))
        avt(ax.data(), &workd[ipntr[1] - 1]);
        // c
        // c           %-----------------------------------------%
        // c           | L O O P   B A C K to call DSAUPD again. |
        // c           %-----------------------------------------%
        // c
      } while (ido == -1 || ido == 1);

      // c     %----------------------------------------%
      // c     | Either we have convergence or there is |
      // c     | an error.                              |
      // c     %----------------------------------------%
      // c
      if (info < 0) {
        // c
        // c        %--------------------------%
        // c        | Error message. Check the |
        // c        | documentation in DSAUPD. |
        // c        %--------------------------%
        // c
        std::cout << "[ArpackSvdSolver::solve] Error with _saupd, info = "
                  << info << std::endl;
        std::cout << "[ArpackSvdSolver::solve] Check documentation in _saupd "
                  << std::endl;
      } else {
        // c
        // c        %--------------------------------------------%
        // c        | No fatal errors occurred.                  |
        // c        | Post-Process using DSEUPD.                 |
        // c        |                                            |
        // c        | Computed singular values may be extracted. |
        // c        |                                            |
        // c        | Singular vectors may also be computed now  |
        // c        | if desired.  (indicated by rvec = .true.)  |
        // c        |                                            |
        // c        | The routine DSEUPD now called to do this   |
        // c        | post processing                            |
        // c        %--------------------------------------------%
        // c
        rvec = 1;
        std::string howmny = "All";

        dseupd_(&rvec, howmny.c_str(), select.data(), s.data(), v.data(), &ldv,
                &sigma, bmat.c_str(), &n, which.c_str(), &nev, &tol,
                resid.data(), &ncv, v.data(), &ldv, iparam.data(), ipntr.data(),
                workd.data(), workl.data(), &lworkl, &ierr);
        // c
        // c        %-----------------------------------------------%
        // c        | Singular values are returned in the first     |
        // c        | column of the two dimensional array S         |
        // c        | and the corresponding right singular vectors  |
        // c        | are returned in the first NEV columns of the  |
        // c        | two dimensional array V as requested here.    |
        // c        %-----------------------------------------------%
        // c
        if (ierr != 0) {
          // c
          // c           %------------------------------------%
          // c           | Error condition:                   |
          // c           | Check the documentation of DSEUPD. |
          // c           %------------------------------------%
          // c
          std::cout << "[ArpackSvdSolver::solve] Error with _seupd, info = "
                    << ierr << std::endl;
          std::cout
            << "[ArpackSvdSolver::solve] Check the documentation of _seupd. "
            << std::endl;
        } else {
          nconv = iparam[5 - 1];
          for (int j = 1; j <= nconv; j++) {
            // s(j,1) = sqrt(s(j,1))
            s[j - 1] = std::sqrt(s[j - 1]);
            // c
            // c              %-----------------------------%
            // c              | Compute the left singular   |
            // c              | vectors from the formula    |
            // c              |                             |
            // c              |     u = Av/sigma            |
            // c              |                             |
            // c              | u should have norm 1 so     |
            // c              | divide by norm(Av) instead. |
            // c              %-----------------------------%
            // c
            // call av(m, n, v(1,j), ax)
            av(&v[ldv * (j - 1)], ax.data());
            // call dcopy(m, ax, 1, u(1,j), 1)
            std::copy(ax.data(), ax.data() + m, &u[ldu * (j - 1)]);
            // temp = one/dnrm2(m, u(1,j), 1)
            temp = 1.0 / dnrm2_wrapper(m, &u[ldu * (j - 1)], 1);
            // call dscal(m, temp, u(1,j), 1)
            dscal_wrapper(m, temp, &u[ldu * (j - 1)], 1);
            // c
            // c              %---------------------------%
            // c              |                           |
            // c              | Compute the residual norm |
            // c              |                           |
            // c              |   ||  A*v - sigma*u ||    |
            // c              |                           |
            // c              | for the NCONV accurately  |
            // c              | computed singular values  |
            // c              | and vectors.  (iparam(5)  |
            // c              | indicates how many are    |
            // c              | accurate to the requested |
            // c              | tolerance).               |
            // c              | Store the result in 2nd   |
            // c              | column of array S.        |
            // c              %---------------------------%
            // c
            // call daxpy(m, -s(j,1), u(1,j), 1, ax, 1)
            daxpy_wrapper(m, -s[j - 1], &u[ldu * (j - 1)], 1, ax.data(), 1);
            // s(j,2) = dnrm2(m, ax, 1)
            s[j + ncv - 1] = dnrm2_wrapper(m, ax.data(), 1);
          }
          // ----- CONVERGENCE INFORMATION
          // --------------------------------------
          // c
          // c           %-------------------------------%
          // c           | Display computed residuals    |
          // c           %-------------------------------%
          // c
          //  call dmout(6, nconv, 2, s, maxncv, -6,
          //&                'Singular values and direct residuals')
          if (dbg) {
            std::cout << "[ArpackSvdSolver::solve] Computed sing. values and "
                         "residuals: "
                      << std::endl;
            for (int i = 0; i < nconv; i++) {
              std::cout << s[i] << " " << s[ncv + i] << std::endl;
            }
          }
        }
        // c
        // c        %------------------------------------------%
        // c        | Print additional convergence information |
        // c        %------------------------------------------%
        // c
        if (dbg) {
          if (info == 1) {
            std::cout << "[ArpackSvdSolver::solve] Maximum number of "
                         "iterations reached."
                      << std::endl;
          } else if (info == 3) {
            std::cout << "[ArpackSvdSolver::solve] No shifts could be applied "
                         "during implicit"
                      << " Arnoldi update, try increasing NCV." << std::endl;
          }

          std::cout << "[ArpackSvdSolver::solve] _SVD " << std::endl;
          std::cout << "[ArpackSvdSolver::solve] Size of the matrix is " << n
                    << std::endl;
          std::cout << "[ArpackSvdSolver::solve] The number of Ritz values "
                       "requested is "
                    << nev << std::endl;
          std::cout << "[ArpackSvdSolver::solve] The number of Arnoldi vectors "
                       "generated"
                    << " (NCV) is " << ncv << std::endl;
          std::cout << "[ArpackSvdSolver::solve] What portion of the spectrum: "
                    << which << std::endl;
          std::cout << "[ArpackSvdSolver::solve] The number of converged Ritz "
                       "values is "
                    << nconv << std::endl;
          std::cout
            << "[ArpackSvdSolver::solve] The number of Implicit Arnoldi update"
            << " iterations taken is " << iparam[2] << std::endl;
          std::cout << "[ArpackSvdSolver::solve] The number of OP*x is "
                    << iparam[8] << std::endl;
          std::cout << "[ArpackSvdSolver::solve] The convergence criterion is "
                    << tol << std::endl;
        }
      }
      // c
      // c     %-------------------------%
      // c     | Done with program dsvd. |
      // c     %-------------------------%
      // c

      // Arpack orders singular values and corresponding vectors in ascending
      // order to conform with ITensor, its necessary to reverse the order

      // reverse singular values
      for (int i = 0; i < nconv; i++) {
        D[i] = s[nconv - i - 1];
      }

      // reverse U
      std::vector<double> tmp(ldu);
      for (int j = 0; j < nconv / 2; j++) {
        std::copy(U.data() + ldu * j, U.data() + ldu * (j + 1), tmp.data());
        std::copy(U.data() + ldu * (nconv - j - 1),
                  U.data() + ldu * (nconv - j), U.data() + ldu * j);
        std::copy(tmp.data(), tmp.data() + ldu,
                  U.data() + ldu * (nconv - j - 1));
      }

      // copy and reverse v to V
      for (int j = 0; j < nconv; j++) {
        std::copy(v.data() + ldv * j, v.data() + ldv * (j + 1),
                  V.data() + ldv * (nconv - j - 1));
      }

#  ifdef CHKSVD
      checksvd(M, U, D, V);
#  endif
      return;
    }

    // No complex implementation fall back to ITensor
    // template<>
    void solve(MatRefc<Cplx> const& M,
               MatRef<Cplx> const& U,
               VectorRef const& D,
               MatRef<Cplx> const& V,
               Args const& args) {
      SvdSolver::solve(M, U, D, V, args);
    }

    static std::unique_ptr<ArpackSvdSolver> create() {
      return std::unique_ptr<ArpackSvdSolver>(new ArpackSvdSolver());
    }
  };

}  // namespace itensor

#endif

#endif