#include "p-ipeps/config.h"
#include "p-ipeps/linalg/arpack-rcdn.h"
DISABLE_WARNINGS
#include "itensor/all.h"
ENABLE_WARNINGS
	
struct DiagMVP {
	int id;

	DiagMVP(int iid) : id(iid) {}

	void operator() (double const* const x, double* const y) {
		for (int i = 0; i < 1000; ++i) {
			y[i] = static_cast<double>(i + 1) * x[i];
		}
	}
};

namespace itensor {
struct DiagMVP_itensor {
	int N;

	DiagMVP_itensor(int nn) : N(nn) {}

	void operator() (double const* const x, double* const y) {
		
		auto dummy = Index("dummy",1);
		auto i  = Index("i",N);
		auto ip = prime(i);

		// copy x
		std::vector<double> cpx(N); 
		std::copy(x, x+N, cpx.data());

		//auto vecRefX = makeVecRef(cpx.data(),cpx.size()); 

		auto isX = IndexSet(i);
		auto X = ITensor(isX,Dense<double>(std::move(cpx)));
		//auto Y = ITensor({ip},Dense<double>(std::move(y)));
		
		auto A = ITensor(ip,i);
		for (int j=1;j<=N;j++) A.set(ip(j),i(j),10.0 - (j-1)); 

		auto Y = A * X;
		Y.scaleTo(1.0);

		auto extractReal = [](Dense<Real> const& d) {
			return d.store;
		};

		auto yData = applyFunc(extractReal,Y.store());
		std::copy(yData.data(), yData.data()+N, y);
	}
};
}

int main() {
	// DiagMVP dmvp(0);
	// ARDNS<DiagMVP> ardns(dmvp);

	itensor::DiagMVP_itensor dmvp(10);
	ARDNS<itensor::DiagMVP_itensor> ardns(dmvp);


	ardns.real_nonsymm_runner(10, 2, 5, 0.0, 10 * 10);

	// Funcd funcd(1);
	// Linemethod<Funcd> lm(funcd);

	return 0;
}
