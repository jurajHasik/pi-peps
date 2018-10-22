#ifndef __ARPACK_RCI_H_
#define __ARPACK_RCI_H_

#include "arpack.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <vector>

#include "debug_c.hpp"  // debug arpack.
#include "stat_c.hpp"   // arpack statistics.

#ifndef BLASINT
#define BLASINT int
#endif

template<class T>
struct ARDNS {
	T & mvp;

	ARDNS(T & mmvp) : mvp(mmvp) {}

	void real_nonsymm_runner(
	 	BLASINT const N, BLASINT const nev, BLASINT const max_ncv, 
	 	double const tol, BLASINT const maxIter
	) {
		BLASINT const ncv = std::max(max_ncv, 2 * nev + 1);
		BLASINT const ldv = N;
		BLASINT const lworkl = 3 * (ncv * ncv) + 6 * ncv;

		std::cout<<"N "<< N <<"\n"
			<<"nev "<< nev <<"\n"
			<<"tol "<< tol <<"\n"
			<<"maxIter "<< maxIter <<"\n"
			<<"ncv "<< ncv <<"\n"
			<<"ldv "<< ldv <<"\n"
			<<"lworkl "<< lworkl <<"\n";

		  // ***** Local Arrays ********************************************** 
		  // dnaupd
		  std::vector<double> resid(N);
		  std::vector<double> V(ncv * N);
		  std::vector<double> workd(3 * N, 0.0);
		  std::vector<double> workl(lworkl, 0.0);
		  std::array<BLASINT, 11> iparam{};
		  std::array<BLASINT, 14> ipntr{};
		  

		  // ***** Local Scalars ********************************************* 
		  iparam[0] = 1;      // use exact shifts
		  iparam[2] = maxIter; // maximal number of iterations
		  iparam[3] = 1;      
		  iparam[4] = 0;      // number of ev found by arpack.
		  iparam[6] = 1;      // mode 1 of dnaupd (???) related to use of exact shifts iparam[0] = 1

		  BLASINT info = 0, ido = 0;

		  while (ido != 99) {
		    // arpack::saupd(ido, arpack::bmat::identity, N,
		    //               arpack::which::largest_magnitude, nev, tol, resid.data(), ncv,
		    //               V.data(), ldv, iparam.data(), ipntr.data(), workd.data(),
		    //               workl.data(), lworkl, info);

		    arpack::naupd(ido, 
		        arpack::bmat::identity,           // solve Ax = lambda Bx where B = Identity
		        N,                                // linear size of A
		        arpack::which::largest_magnitude, // solve for largest eigenvalues
		        nev, 
		        tol, 
		        resid.data(), 
		        ncv,
		        V.data(), 
		        ldv, 
		        iparam.data(), 
		        ipntr.data(), 
		        workd.data(), 
		        workl.data(), 
		        lworkl, 
		        info 
		      );

		    // perform matrix vector multiplication Ax -> y  
		    mvp(
		        &(workd[ipntr[0] - 1]), // = x
		        &(workd[ipntr[1] - 1])  // = y
		      );
		  }

		  // check number of ev found by arpack.
		  if (iparam[4] != nev || info != 0) {
		  	std::cout<<"[arpack::dnaupd] iparam[4] = " << iparam[4] << std::endl;
		    std::cout<<"[arpack::dnaupd] info=" << info << std::endl;
		    //throw std::domain_error("Error inside ARPACK routines");
		  } else {
		    std::cout<<"Processing result of [arpack::dnaupd]"<<std::endl;
		  }

		  // ***** dnaupd exited succesfully, post-process result ************
		  BLASINT const ldz = N + 1;

		  std::vector<double> dr(nev+1);
		  std::vector<double> di(nev+1);
		  std::vector<double> workev(3 * ncv);
		  std::vector<double> z((N + 1) * (nev + 1));

		  std::vector<int> select(ncv);
		  bool const rvec = false; // Extract eigenvectors Y/N ?
		  
		  double const sigmar = 0.0;
		  double const sigmai = 0.0;

		  arpack::neupd(
		      rvec, 
		      arpack::howmny::ritz_vectors, 
		      select.data(), 
		      dr.data(),                      // first column of d contains real part of eigenvalues
		      di.data(),                      // second column of d contains imag part of eigenvalues
		      z.data(), 
		      ldz, 
		      sigmar,                           //
		      sigmai,                           //
		      workev.data(), 
		      arpack::bmat::identity,           // process for B = Identity
		      N,
		      arpack::which::largest_magnitude, // process for largest eigenvalues
		      nev, 
		      tol, 
		      resid.data(), 
		      ncv,
		      V.data(), 
		      ldv, 
		      iparam.data(), 
		      ipntr.data(), 
		      workd.data(),
		      workl.data(), 
		      lworkl, 
		      info
		    );

		  if ( info != 0 ) {
		    std::cout<<"[arpack::dneupd] info=" << info << std::endl;
		    throw std::domain_error("Error in dneupd");
		  } else {
		    std::cout<<"Processing done [arpack::dneupd]"<<std::endl;
		  }

		  for (int i = 0; i < nev; ++i) {
		    std::cout << dr[i] <<" "<< di[i] <<"i"<< "\n";
		  }
		  std::cout << "------\n";
	}
};

#endif